# Instructions

Setup:

```
pip install -r requirements.txt
```

reward(board_state, model_response) takes in the board state, model response, parses it to extract the move (in \\boxed{}, like \\boxed{Nc3}), and returns the change in win probability from making that move. 

Data should be formatted as a list of strings containing the board positions. 

When annotating positions mention strategic / tactical moves, come up with a few candidate moves, and eliminate some of them before picking the best one. They should be formatted as [{"role": "user", "content": BOARD_POSITION}, {"role": "assistant", "content": ANALYSIS_GOES_HERE}]

