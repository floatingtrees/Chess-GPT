# Instructions

Setup:

```
pip install -r requirements.txt
```

reward(board_state, model_response) takes in the board state, model response, parses it to extract the move (in \\boxed{}, like \\boxed{Nc3}), and returns the change in win probability from making that move. Changes to this function should be made in reward.py

Data should be formatted as a list of strings containing the board positions. Work should happen in process_data.py, and stored as PGN. 

When annotating positions mention strategic / tactical moves, come up with a few candidate moves, and eliminate some of them before picking the best one. Try to be detailed. You should put your analysis in `<think> content </think>` tags. They should be formatted as a list of [{"role": "user", "content": BOARD_POSITION}, {"role": "assistant", "content": ANALYSIS_GOES_HERE}]. BOARD_POSITION can be in PGN position. Annotations can be put in analysis.json. 

For the person figuring out/converting PGN to optimal formats for the model, work in preprocess_position.py. 

Try to stay in your files to avoid merge conflicts, and use python3.12 if possible. 