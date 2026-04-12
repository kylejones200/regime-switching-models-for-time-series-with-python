# Regime Switching Models for Time Series Analysis in Python

We are looking for patterns in time series data across different time intervals. For example, stock market prices might show high volatility during crises and low volatility during stable periods. Regime switching models help identify this behavior by assuming that the time series switches between distinct \"regimes,\" each governed by its own statistical properties.

In this article, we'll explore regime switching models, their applications, and how to implement them in Python using the statsmodels library.

## What Are Regime Switching Models?

Regime switching models, introduced by James Hamilton in 1989, capture structural changes in time series data by allowing transitions between different states or regimes. These models are particularly useful in:

- **Economics:** Modeling recessions and expansions.

- **Finance:** Analyzing bull and bear markets.

- **Energy:** Detecting shifts in demand or production trends.

- **Weather:** Capturing transitions between climatic states.

The most common type of regime switching model is the Markov Switching Model, where regime transitions follow a Markov process.

## Markov Chains and State-Dependent Probability

Markov switching models work like a storyteller who knows multiple versions of the same story. Imagine an economy that switches between boom and bust periods. At any moment, the economy follows specific patterns depending on whether it's in a boom (State A) or bust (State B).

The magic happens through Markov chains, where the next state only depends on the current state, not the past. Think of it like a weather forecast. Tomorrow's weather depends mostly on today's conditions, not what happened last week. The probability of switching between states (transition probability) is fixed for each current state. For example, during a boom, there might be a 90% chance of staying in boom and 10% chance of switching to bust.

Within each state, outcomes follow state-specific probability distributions (state-dependent probability). During booms, growth might average 3% with low volatility, while during busts, it might average -1% with high volatility. The model combines these pieces - regime identification, transition probabilities, and state-specific behavior - to capture complex patterns in data.

The statsmodels library provides a convenient implementation of Markov Switching models through its `MarkovRegression` and `MarkovAutoregression` classes. Let's walk through an example.

We'll generate synthetic data with two regimes: high and low volatility.

    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    import seaborn as sns
    from scipy import stats

    # Generate and prepare data
    np.random.seed(42)
    n = 500
    regimes = np.random.choice([0, 1], size=n, p=[0.7, 0.3])
    data = np.array(np.random.normal(0, np.where(regimes == 0, 1, 5)))

    # 2. Create a DataFrame with all the information
    df = pd.DataFrame({
        'Data': data,
        'True_Regime': regimes,
        'Time': range(n)
    })

    # Fit a Markov switching model
    model = MarkovRegression(data, k_regimes=2, trend='c', switching_variance=True)
    result = model.fit()
    print(result.summary())

    print("\nTransition Matrix:")
    print(result.regime_transition)

Now we have actual and predicted values for each point.

    # Add predicted probabilities and regimes to DataFrame
    df['Predicted_Prob_High'] = result.smoothed_marginal_probabilities[:, 1]
    df['Predicted_Regime'] = np.argmax(result.smoothed_marginal_probabilities, axis=1)

    # Plot 1: Original Data with Regime Highlighting
    plt.figure(figsize=(12, 6))
    for regime in [0, 1]:
        mask = df['True_Regime'] == regime
        plt.scatter(df[mask]['Time'], df[mask]['Data'],
                    label=f"Regime {regime}", alpha=0.6)
    plt.title("Original Data with True Regimes")
    plt.legend()
    plt.savefig('original_data_regimes.png')
    plt.close()

    # Plot 2: Predicted vs True Regimes
    plt.figure(figsize=(12, 6))
    plt.plot(df['True_Regime'], label='True Regime', alpha=0.6)
    plt.plot(df['Predicted_Regime'], label='Predicted Regime', alpha=0.6)
    plt.title("True vs Predicted Regimes")
    plt.legend()
    plt.savefig('true_vs_predicted_regimes.png')
    plt.close()

    # Plot 3: Density Plot for Each Regime
    plt.figure(figsize=(12, 6))
    for regime in [0, 1]:
        sns.kdeplot(data=df[df['True_Regime'] == regime]['Data'],
                    label=f"Regime {regime}")
    plt.title("Density Distribution by Regime")
    plt.legend()
    plt.savefig('density_distribution.png')
    plt.close()

    # Plot 4: Transition Probability Matrix Heatmap
    plt.figure(figsize=(8, 6))
    transition_matrix = result.regime_transition.reshape(2, 2)
    sns.heatmap(transition_matrix, annot=True, cmap='coolwarm')
    plt.title("Transition Probability Matrix")
    plt.xlabel("To Regime")
    plt.ylabel("From Regime")
    plt.savefig('transition_matrix.png')
    plt.close()

    # Plot 5: Model Performance Metrics
    plt.figure(figsize=(8, 6))
    confusion_matrix = pd.crosstab(df['True_Regime'], df['Predicted_Regime'])
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix: True vs Predicted Regimes")
    plt.xlabel("Predicted Regime")
    plt.ylabel("True Regime")
    plt.savefig('confusion_matrix.png')
    plt.close()

Like other time series models, we need to ensure the data is stationary before fitting a regime switching model. We can test for stationarity with Dickey-Fuller or KPSS. We also need to define the number of regimes we believe are in the data. We can do that based on domain knowledge or using a threshold like maximizing the Akaike Information Criterion (AIC).

Regime switching models provide a flexible framework for analyzing time series with structural changes. They can help us find insights that traditional models might miss. Using Python and statsmodels, you can efficiently implement these models and adapt them to your specific needs.

Regime-specific models improve prediction accuracy for forecasting. They can be used to spot significant shifts in time series behavior for change point detection.

Regime switching can also be used for feature engineering for time series classification or clustering tasks to prepare data for further ML work.

## Key Takeaways

- **Economics:** Modeling recessions and expansions.
- **Finance:** Analyzing bull and bear markets.
- **Energy:** Detecting shifts in demand or production trends.
- **Weather:** Capturing transitions between climatic states.
