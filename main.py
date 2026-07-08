from models.feature_engine import FeatureEngine
from models.hypothesis_engine import HypothesisEngine
from models.validation_engine import ValidationEngine
from models.experiment_manager import ExperimentManager
from models.experiment_runner import ExperimentRunner
from models.dataset_splitter import split_dataframe


FEATURE_RULES = {
    "body_ratio": (">", 0.6),
    "upper_wick_ratio": (">", 0.2),
    "lower_wick_ratio": (">", 0.2),
    "close_pos_in_range": (">", 0.5),
    "atr_14": (">", 0.0),
    "momentum_3": (">", 0.0),
}


def main() -> None:
    feature_engine = FeatureEngine()
    experiment_manager = ExperimentManager()
    hypothesis_engine = HypothesisEngine()
    validator = ValidationEngine()
    runner = ExperimentRunner()

    try:
        df = feature_engine.run()
        print(f"Loaded and engineered features: {len(df):,} rows, {len(df.columns):,} columns")

        split = split_dataframe(df, train_ratio=0.70, validation_ratio=0.15)
        print(
            f"Split -> train: {len(split.train):,}, validation: {len(split.validation):,}, test: {len(split.test):,}"
        )

        generated = list(hypothesis_engine.generate(FEATURE_RULES, max_features=3))
        print(f"Generated hypotheses: {len(generated):,}")

        accepted = 0
        for hyp in generated:
            train_result = runner.evaluate(split.train, hyp)
            validation_result = runner.evaluate(split.validation, hyp)
            test_result = runner.evaluate(split.test, hyp)

            validation = validator.evaluate(
                train_winrate=train_result.winrate,
                validation_winrate=validation_result.winrate,
                test_winrate=test_result.winrate,
                occurrence=validation_result.occurrence,
            )

            exp = experiment_manager.create(
                hypothesis=hyp.id,
                parameters={
                    "signature": hyp.signature,
                    "conditions": [(c.feature, c.operator, c.value) for c in hyp.conditions],
                },
                dataset="feature_frame",
            )
            exp.status = "PASS" if validation.passed else "REJECT"
            exp.train_win = train_result.winrate
            exp.validation_win = validation_result.winrate
            exp.test_win = test_result.winrate
            experiment_manager.save(exp)

            if validation.passed:
                accepted += 1

        print(f"Accepted hypotheses: {accepted}")
    finally:
        feature_engine.close()


if __name__ == "__main__":
    main()
