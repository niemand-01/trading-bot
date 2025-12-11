# Model Evaluation Guide

## Quick Start

### Evaluate a Single Model
```bash
python eval.py data/GOOG_2019.csv --model-name model_debug_50
```

### Evaluate All Models
```bash
python eval.py data/GOOG_2019.csv
```

## Available Models

Based on your `models/` directory, you have:

### Saved Model Directories (New Format):
- `model_debug_10` - Saved at episode 10
- `model_debug_20` - Saved at episode 20
- `model_debug_30` - Saved at episode 30
- `model_debug_40` - Saved at episode 40
- `model_debug_50` - Saved at episode 50 (final)

### Old Model Files:
- `model_t-dqn_GOOG_10` - Target DQN, 10 episodes
- `model_dqn_GOOG_50` - Vanilla DQN, 50 episodes
- `model_double-dqn_GOOG_50` - Double DQN, 50 episodes

## Evaluation Commands

### 1. Evaluate Latest Model (Episode 50)
```bash
python eval.py data/GOOG_2019.csv --model-name model_debug_50
```

### 2. Evaluate Specific Episode
```bash
# Episode 10
python eval.py data/GOOG_2019.csv --model-name model_debug_10

# Episode 20
python eval.py data/GOOG_2019.csv --model-name model_debug_20

# Episode 30
python eval.py data/GOOG_2019.csv --model-name model_debug_30

# Episode 40
python eval.py data/GOOG_2019.csv --model-name model_debug_40
```

### 3. Evaluate All Models (Compare Performance)
```bash
python eval.py data/GOOG_2019.csv
```
This will evaluate all model files in the `models/` directory and show which performs best.

### 4. Evaluate with Debug Output (See Trading Decisions)
```bash
python eval.py data/GOOG_2019.csv --model-name model_debug_50 --debug
```
Shows detailed BUY/SELL decisions during evaluation.

### 5. Evaluate on Different Test Data
```bash
# Test on 2019 data
python eval.py data/GOOG_2019.csv --model-name model_debug_50

# Test on 2018 data (validation set)
python eval.py data/GOOG_2018.csv --model-name model_debug_50

# Test on other stocks
python eval.py data/AAPL.csv --model-name model_debug_50
```

### 6. Custom Window Size
```bash
python eval.py data/GOOG_2019.csv --model-name model_debug_50 --window-size 10
```

## Understanding the Output

### Example Output:
```
model_debug_50: +$355.88
```

This means:
- **Model**: `model_debug_50`
- **Profit**: +$355.88 (positive = profit, negative = loss)
- **Test Data**: GOOG_2019.csv (2019 stock data)

### Output Format:
- `+$XXX.XX` = Profit (good!)
- `-$XXX.XX` = Loss (bad)
- `USELESS` = Model didn't trade meaningfully (profit ≈ trivial baseline)

## Comparing Models

### Compare All Models at Once:
```bash
python eval.py data/GOOG_2019.csv
```

Output will show:
```
model_debug_10: +$123.45
model_debug_20: +$234.56
model_debug_30: +$345.67
model_debug_40: +$456.78
model_debug_50: +$355.88
```

**Best Model**: The one with highest profit (or lowest loss if all negative)

### Compare Different Strategies:
```bash
# Target DQN
python eval.py data/GOOG_2019.csv --model-name model_t-dqn_GOOG_10

# Vanilla DQN
python eval.py data/GOOG_2019.csv --model-name model_dqn_GOOG_50

# Double DQN
python eval.py data/GOOG_2019.csv --model-name model_double-dqn_GOOG_50
```

## Evaluation Process

### What Happens During Evaluation:

1. **Load Model**: Loads the trained neural network
2. **Load Test Data**: Reads stock price data from CSV
3. **Run Trading Simulation**:
   - Agent observes each day's price pattern
   - Selects best action (BUY/SELL/HOLD) - **no exploration**
   - Executes trades and tracks profit/loss
4. **Calculate Results**: Sums all profits from completed trades
5. **Display**: Shows total profit/loss

### Key Differences from Training:

| Aspect | Training | Evaluation |
|--------|----------|------------|
| **Exploration** | Yes (epsilon-greedy) | No (best action only) |
| **Model Updates** | Yes (learns) | No (frozen) |
| **Purpose** | Learn patterns | Test learned knowledge |
| **Random Actions** | Yes (exploration) | No (deterministic) |

## Best Practices

### 1. Evaluate on Unseen Data
- Use data **after** training period
- Example: Train on 2010-2017, evaluate on 2019
- Tests generalization ability

### 2. Compare Multiple Episodes
- Evaluate models from different training stages
- See if performance improves with more training
- Identify best stopping point

### 3. Test on Multiple Datasets
- Evaluate on different stocks
- Test on different time periods
- Measure robustness

### 4. Use Debug Mode for Analysis
```bash
python eval.py data/GOOG_2019.csv --model-name model_debug_50 --debug
```
Shows:
- When agent buys
- When agent sells
- Profit/loss per trade
- Helps understand trading strategy

## Example Evaluation Workflow

### Step 1: Evaluate All Models
```bash
python eval.py data/GOOG_2019.csv
```

### Step 2: Identify Best Model
Look for highest profit in output.

### Step 3: Detailed Analysis of Best Model
```bash
python eval.py data/GOOG_2019.csv --model-name model_debug_50 --debug
```

### Step 4: Test on Other Data
```bash
# Test on validation data
python eval.py data/GOOG_2018.csv --model-name model_debug_50

# Test on other stocks
python eval.py data/AAPL.csv --model-name model_debug_50
```

## Troubleshooting

### Model Not Found
```
Error: Could not find model
```
**Solution**: Check model name matches exactly (case-sensitive)
```bash
# List available models
ls models/
```

### Wrong Window Size
```
Error: Model expects different input shape
```
**Solution**: Use same window size as training
```bash
# If trained with window-size=10 (default)
python eval.py data/GOOG_2019.csv --model-name model_debug_50 --window-size 10
```

### GPU Issues
If GPU errors occur, force CPU:
```python
# In eval.py, change line 63:
switch_k_backend_device(use_gpu=False)  # Use CPU
```

## Expected Results

Based on README, good models should achieve:
- **Validation (2018)**: ~$863 profit
- **Test (2019)**: ~$1,141 profit

Your results may vary based on:
- Training duration
- Hyperparameters
- Market conditions
- Random seed

## Next Steps

After evaluation:
1. **Visualize Results**: Use `visualize.ipynb` to see trading decisions
2. **Compare Strategies**: Test different DQN variants
3. **Optimize**: Adjust hyperparameters based on results
4. **Deploy**: Use best model for live trading (with caution!)

