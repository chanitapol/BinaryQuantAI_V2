from models.feature_engine import FeatureEngine


def main() -> None:
    engine = FeatureEngine()
    try:
        df = engine.run()
        print(f"Loaded and engineered features: {len(df):,} rows, {len(df.columns):,} columns")
        print(df.head(3).to_string(index=False))
    finally:
        engine.close()


if __name__ == "__main__":
    main()
