from models.feature_engine import FeatureEngine
from models.hypothesis_engine import HypothesisEngine
from models.validation_engine import ValidationEngine
from models.experiment_manager import ExperimentManager
from models.experiment_runner import ExperimentRunner
from models.dataset_splitter import split_dataframe
from models.ranking_engine import RankingEngine


def main() -> None:
    feature_engine = FeatureEngine()
    experiment_manager = ExperimentManager()
    hypothesis_engine = HypothesisEngine()
    validator = ValidationEngine()
    runner = ExperimentRunner()
    ranking_engine = RankingEngine(payout=0.80)

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

            exp = experiment_manager.create(
                hypothesis=hyp.id,
                parameters={
                    "signature": hyp.signature,
                    "conditions": [(c.feature, c.operator, c.value) for c in hyp.conditions],
                    "direction": hyp.direction,
                },
                dataset="feature_frame",
            )
            exp.status = "PASS" if validation.passed else "REJECT"
            exp.train_win = train_result.winrate
            exp.validation_win = validation_result.winrate
            exp.test_win = test_result.winrate
            experiment_manager.save(exp)

            results.append(
                {
                    "hypothesis_id": hyp.id,
                    "train_winrate": train_result.winrate,
                    "validation_winrate": validation_result.winrate,
                    "test_winrate": test_result.winrate,
                    "occurrence": validation_result.occurrence,
                    "gap": validation.gap,
                    "status": exp.status,
                    "reason": validation.reason,
                    "score": score,
                    "expectancy": expectancy,
                    "confidence": confidence,
                    "stability": stability,
                }
            )

        import pandas as pd

        results_df = pd.DataFrame(results)
        ranked = ranking_engine.rank_rows(results_df)

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
    finally:
        feature_engine.close()


if __name__ == "__main__":
    main()
