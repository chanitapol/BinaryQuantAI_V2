from models.feature_engine import FeatureEngine
from models.hypothesis_engine import HypothesisEngine
from models.validation_engine import ValidationEngine
from models.experiment_runner import ExperimentRunner
from models.dataset_splitter import split_dataframe
from models.ranking_engine import RankingEngine
from models.knowledge_engine import KnowledgeEngine
from models.statistics_engine import StatisticsEngine
from models.evolution_engine import EvolutionEngine

import pandas as pd
import time


def main() -> None:
    feature_engine = FeatureEngine()
    hypothesis_engine = HypothesisEngine()
    validator = ValidationEngine()
    runner = ExperimentRunner()
    ranking_engine = RankingEngine(payout=0.80)
    knowledge = KnowledgeEngine("research.db")
    stats = StatisticsEngine("research.db")
    evolution = EvolutionEngine("research.db")

    start = time.time()
    run_id = 1

    try:
        df = feature_engine.run()
        print("=" * 70)
        print("BinaryQuantAI V2")
        print("=" * 70)
        print(f"Rows      : {len(df):,}")
        print(f"Features  : {len(df.columns):,}")

        split = split_dataframe(df, train_ratio=0.70, validation_ratio=0.15)
        print(
            f"Split -> train: {len(split.train):,}, validation: {len(split.validation):,}, test: {len(split.test):,}"
        )

        hypotheses = hypothesis_engine.generate_from_dataframe(df, max_features=2)
        print(f"Generated hypotheses: {len(hypotheses):,}")

        for hyp in hypotheses:
            knowledge.add_hypothesis(hyp)

        results = []
        for hyp in hypotheses:
            train_result = runner.evaluate(split.train, hyp)
            validation_result = runner.evaluate(split.validation, hyp)
            test_result = runner.evaluate(split.test, hyp)

            validation = validator.evaluate(
                train_winrate=train_result.winrate,
                validation_winrate=validation_result.winrate,
                test_winrate=test_result.winrate,
                occurrence=validation_result.occurrence,
            )

            score, expectancy, confidence, stability = ranking_engine.score(
                train_winrate=train_result.winrate,
                validation_winrate=validation_result.winrate,
                test_winrate=test_result.winrate,
                occurrence=validation_result.occurrence,
                gap=validation.gap,
            )

            row = {
                "run_id": run_id,
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
                "status": "PASS" if validation.passed else "REJECT",
                "runtime": 0.0,
            }
            results.append(row)
            knowledge.add_experiment(row)

        results_df = pd.DataFrame(results)
        ranked = ranking_engine.rank_rows(results_df)

        for _, row in ranked.iterrows():
            knowledge.add_ranking(
                run_id=run_id,
                hypothesis_id=row["hypothesis_id"],
                rank=int(row["rank"]),
                score=float(row["score"]),
            )

        knowledge.flush()

        accepted = int((ranked["status"] == "PASS").sum())
        print(f"Accepted hypotheses: {accepted}/{len(ranked)}")

        print("\nTop 20 hypotheses")
        print("-" * 120)
        for _, row in ranked.head(20).iterrows():
            print(
                f"{row['rank']:4d} | {row['hypothesis_id']} | {row['status']:7s} | "
                f"score={row['score']:.4f} | win={row['validation_winrate']:.4f} | "
                f"exp={row['expectancy']:.4f} | conf={row['confidence']:.4f} | "
                f"stab={row['stability']:.4f} | occ={int(row['occurrence'])} | gap={row['gap']:.4f}"
            )

        print("\nTop Features")
        print("-" * 120)
        top_features = stats.top_features(10)
        if not top_features.empty:
            print(top_features.to_string(index=False))
        else:
            print("No feature statistics yet.")

        print("\nBest Thresholds")
        print("-" * 120)
        best_thresholds = stats.best_thresholds(10)
        if not best_thresholds.empty:
            print(best_thresholds.to_string(index=False))
        else:
            print("No threshold statistics yet.")

        proposals = evolution.proposals_as_hypotheses(top_n=10)
        print("\nEvolution Proposals")
        print("-" * 120)
        for p in proposals[:10]:
            print(f"{p.id} | {p.direction} | {[(c.feature, c.operator, c.value) for c in p.conditions]} | {p.signature}")

        print(f"\nRuntime : {time.time() - start:.2f} sec")
    finally:
        evolution.close()
        stats.close()
        knowledge.close()
        feature_engine.close()


if __name__ == "__main__":
    main()
