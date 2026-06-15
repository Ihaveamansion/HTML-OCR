Use python to take varied color, position, scaling, and font PNG screenshots of html text
Convert and scale those PNGs to rgb 100x100x3 and store them as npz arrays
All the while, keep track of all images/npy label pairs with a .json dictionary
Split all data 80 train 10 val 10 test
Feed npy/label pairs into network to train,
Automatically create error graphs in validation, and subgraphs for specific categories, like is_text_darker, string lengths, ratios