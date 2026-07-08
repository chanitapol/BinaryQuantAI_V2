from models.feature_engine import FeatureEngine
from models.hypothesis_engine import HypothesisEngine
from models.validation_engine import ValidationEngine
from models.experiment_manager import ExperimentManager


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

    try:
        df = feature_engine.run()
        print(f"Loaded and engineered features: {len(df):,} rows, {len(df.columns):,} columns")

        generated = list(hypothesis_engine.generate(FEATURE_RULES, max_features=3))
        print(f"Generated hypotheses: {len(generated):,}")

        # Placeholder evaluation loop wired to the new pipeline.
        # The next commit will replace this with real backtesting logic over feature slices.
        accepted = 0
        for hyp in generated[:25]:
            # Dummy pass-through metrics for now so the pipeline is wired end-to-end.
            result = validator.evaluate(winrate=0.50, occurrence=1000)
            exp = experiment_manager.create(hypothesis=hyp.id, parameters={"signature": hyp.signature}, dataset="feature_frame")
            exp.status = "PASS" if result.passed else "REJECT"
            exp.train_win = result.winrate
            exp.validation_win = result.winrate
            exp.test_win = result.winrate
            experiment_manager.save(exp)
            if result.passed:
                accepted += 1

        print(f"Accepted hypotheses: {accepted}")
    finally:
        feature_engine.close()


if __name__ == "__main__":
    main()
