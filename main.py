from models.feature_engine import FeatureEngine
from models.hypothesis_engine import HypothesisEngine
from models.validation_engine import ValidationEngine
from models.experiment_runner import ExperimentRunner
from models.dataset_splitter import split_dataframe
from models.ranking_engine import RankingEngine
from models.knowledge_engine import KnowledgeEngine
from models.statistics_engine import StatisticsEngine
from models.evolution_engine import EvolutionEngine
from models.audit_engine import AuditEngine

import pandas as pd
import time


def _evaluate_hypotheses(hypotheses, split, runner, validator, ranking_engine, knowledge, run_id: int, generation: int):
    results = []
    mode = "exploration" if generation == 0 else "production"
    for hyp in hypotheses:
        train_result = runner.evaluate(split.train, hyp)
        validation_result = runner.evaluate(split.validation, hyp)
        test_result = runner.evaluate(split.test, hyp)

        validation = validator.evaluate(
            train_winrate=train_result.winrate,
            validation_winrate=validation_result.winrate,
            test_winrate=test_result.winrate,
            occurrence=validation_result.occurrence,
            mode=mode,
        )

        if validation.status == "PASS":
            print(hyp.id, train_result.winrate, validation_result.winrate, test_result.winrate, validation_result.occurrence)

        score, expectancy, confidence, stability = ranking_engine.score(
            train_winrate=train_result.winrate,
            validation_winrate=validation_result.winrate,
            test_winrate=test_result.winrate,
            occurrence=validation_result.occurrence,
            gap=validation.gap,
        )

        row = {
            "run_id": run_id,
            "generation": generation,
            "hypothesis_id": hyp.id,
            "train_winrate": train_result.winrate,
            "validation_winrate": validation_result.winrate,
            "test_winrate": test_result.winrate,
            "occurrence": validation_result.occurrence,
            "expectancy": expectancy,
            "confidence": confidence,
            "stability": stability,
            "gap": validation.gap,
            "score": score,
            "status": validation.status,
            "reject_reason": validation.reason,
            "runtime": 0.0,
        }
        results.append(row)
        knowledge.add_experiment(row)

    results_df = pd.DataFrame(results)
    ranked = ranking_engine.rank_rows(results_df)

    for _, row in ranked.iterrows():
        knowledge.add_ranking(
            run_id=run_id,
            generation=generation,
            hypothesis_id=row["hypothesis_id"],
            rank=int(row["rank"]),
            score=float(row["score"]),
        )

    knowledge.flush()
    return ranked


def _audit_report(audit_result) -> None:
    print("\nAudit Report")
    print("-" * 120)
    print(f"Passed: {audit_result.passed}")
    if audit_result.issues:
        print("Issues:")
        for issue in audit_result.issues:
            print(f"- {issue}")
    else:
        print("Issues: none")
    print("Metrics:")
    for key, value in audit_result.metrics.items():
        print(f"- {key}: {value}")


def _print_feature_diagnostics(df: pd.DataFrame, hypothesis_engine: HypothesisEngine) -> None:
    diag = hypothesis_engine.diagnostics(df)
    print("\nFeature Diagnostics")
    print("-" * 120)
    print(f"Total numeric features: {diag['total_numeric_features']}")
    print(f"Semantic feature count : {diag['semantic_count']}")
    prefix_counts = diag.get("prefix_counts", {})
    print(
        "Indicator prefixes     : "
        f"RSI={prefix_counts.get('rsi', 0)}, EMA={prefix_counts.get('ema', 0)}, MACD={prefix_counts.get('macd', 0)}, "
        f"BB={prefix_counts.get('bb', 0)}, Stoch={prefix_counts.get('stoch', 0)}, ADX={prefix_counts.get('adx', 0)}, "
        f"Regime={prefix_counts.get('regime', 0)}"
    )
    inventory = hypothesis_engine.feature_inventory(df)
    print("\nTop 20 feature inventory")
    print("-" * 120)
    if not inventory.empty:
        print(inventory.head(20).to_string(index=False))
    else:
        print("No features detected.")


def main() -> None:
    feature_engine = FeatureEngine()
    hypothesis_engine = HypothesisEngine()
    validator = ValidationEngine()
    runner = ExperimentRunner()
    ranking_engine = RankingEngine(payout=0.80, exploration_mode=True)
    knowledge = KnowledgeEngine("research.db")
    stats = None
    evolution = None
    audit_engine = AuditEngine()

    start = time.time()
    run_id = None
    run_completed = False

    try:
        df = feature_engine.run()
        run_id = knowledge.start_run(rows=len(df), features=len(df.columns), notes="BinaryQuantAI V2 research run")

        print("=" * 70)
        print("BinaryQuantAI V2")
        print("=" * 70)
        print(f"Run ID    : {run_id}")
        print(f"Rows      : {len(df):,}")
        print(f"Features  : {len(df.columns):,}")

        _print_feature_diagnostics(df, hypothesis_engine)

        split = split_dataframe(df, train_ratio=0.70, validation_ratio=0.15)
        print(f"Split -> train: {len(split.train):,}, validation: {len(split.validation):,}, test: {len(split.test):,}")

        audit_result = audit_engine.audit(df, split)
        _audit_report(audit_result)

        hypotheses = hypothesis_engine.generate_from_dataframe(df, max_features=2)
        print(f"\nGenerated hypotheses: {len(hypotheses):,}")
        for hyp in hypotheses:
            knowledge.add_hypothesis(hyp)

        ranked = _evaluate_hypotheses(
            hypotheses, split, runner, validator, ranking_engine, knowledge,
            run_id=run_id, generation=0,
        )

        stats = StatisticsEngine("research.db", run_id=run_id, generation=0)
        evolution = EvolutionEngine("research.db")
        evolution.query.set_scope(run_id=run_id, generation=0)

        accepted = int((ranked["status"] == "PASS").sum())
        watched = int((ranked["status"].astype(str).str.startswith("WATCH")).sum())
        print(f"Accepted hypotheses: {accepted}/{len(ranked)}")
        print(f"Watched hypotheses  : {watched}/{len(ranked)}")

        print("\nTop 20 hypotheses")
        print("-" * 120)
        for _, row in ranked.head(20).iterrows():
            reason = row.get("reject_reason", "accepted")
            print(
                f"{row['rank']:4d} | {row['hypothesis_id']} | {row['status']:7s} | "
                f"score={row['score']:.4f} | win={row['validation_winrate']:.4f} | "
                f"exp={row['expectancy']:.4f} | conf={row['confidence']:.4f} | "
                f"stab={row['stability']:.4f} | occ={int(row['occurrence'])} | gap={row['gap']:.4f} | reason={reason}"
            )

        print("\nTop Features (current run, generation 0)")
        print("-" * 120)
        top_features = stats.top_features(10)
        print(top_features.to_string(index=False) if not top_features.empty else "No feature statistics yet.")

        print("\nBest Thresholds (current run, generation 0)")
        print("-" * 120)
        best_thresholds = stats.best_thresholds(10)
        print(best_thresholds.to_string(index=False) if not best_thresholds.empty else "No threshold statistics yet.")

        proposals = evolution.proposals_as_hypotheses(top_n=10)
        print("\nEvolution Proposals")
        print("-" * 120)
        if proposals:
            for p in proposals[:10]:
                print(f"{p.id} | {p.direction} | {[(c.feature, c.operator, c.value) for c in p.conditions]} | {p.signature}")
        else:
            print("No evolution proposals.")

        if proposals:
            for hyp in proposals:
                knowledge.add_hypothesis(hyp)

            evolved_ranked = _evaluate_hypotheses(
                proposals, split, runner, validator, ranking_engine, knowledge,
                run_id=run_id, generation=1,
            )

            accepted_evo = int((evolved_ranked["status"] == "PASS").sum())
            watched_evo = int((evolved_ranked["status"].astype(str).str.startswith("WATCH")).sum())
            print(f"\nEvolution Accepted hypotheses: {accepted_evo}/{len(evolved_ranked)}")
            print(f"Evolution Watched hypotheses  : {watched_evo}/{len(evolved_ranked)}")
            print("\nTop 10 evolution hypotheses")
            print("-" * 120)
            for _, row in evolved_ranked.head(10).iterrows():
                reason = row.get("reject_reason", "accepted")
                print(
                    f"{row['rank']:4d} | {row['hypothesis_id']} | {row['status']:7s} | "
                    f"score={row['score']:.4f} | win={row['validation_winrate']:.4f} | "
                    f"exp={row['expectancy']:.4f} | conf={row['confidence']:.4f} | "
                    f"stab={row['stability']:.4f} | occ={int(row['occurrence'])} | gap={row['gap']:.4f} | reason={reason}"
                )
        else:
            print("\nEvolution Accepted hypotheses: 0/0")
            print("Evolution Watched hypotheses  : 0/0")

        knowledge.finish_run(run_id, status="COMPLETED")
        run_completed = True
        print(f"\nRuntime : {time.time() - start:.2f} sec")
    except Exception as exc:
        if run_id is not None and not run_completed:
            knowledge.finish_run(run_id, status="FAILED", notes=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if evolution is not None:
            evolution.close()
        if stats is not None:
            stats.close()
        knowledge.close()
        feature_engine.close()


if __name__ == "__main__":
    main()