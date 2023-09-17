import random
import tensorflow as tf 
import numpy as np 

def player(prev_play, opponent_history=[]):
    opponent_history.append(prev_play)

    # Strategy 1: Random move
    moves = ["R", "P", "S"]
    random_move = random.choice(moves)

    # Strategy 2: Counter the opponent's last move
    if prev_play == "R":
        counter_move = "P"
    elif prev_play == "P":
        counter_move = "S"
    elif prev_play == "S":
        counter_move = "R"
    else:
        counter_move = random_move

    # Strategy 3: Analyze opponent history and predict the next move
    opponent_moves = opponent_history[-3:]
    predicted_move = predict_next_move(opponent_moves)

    # Select the move based on the strategies
    if predicted_move:
        return predicted_move
    elif random.random() < 0.6:
        return counter_move
    else:
        return random_move

def predict_next_move(opponent_moves):
    #encoding 
    encoding_move={'R': [1, 0, 0], 'P': [0, 1, 0], 'S': [0, 0, 1]}
    encoded_moves=[encoding_move[move] for move in opponent_moves if move]
    
    x=np.array(encoded_moves[:-1])
    y=np.array(encoded_moves[1:])

    model=tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(16,input_shape=(3,),activation='relu'),
            tf.keras.layers.Dense(16,activation='relu'),
            tf.keras.layers.Dense(3,activation='softmax')
        ]
    )

    model.compile(optimizer='adam',loss="categorical_crossentropy")
    model.fit(x,y,epochs=10,verbose=0)

    last_move=opponent_moves[-1] if opponent_moves else ''
    if last_move:
        last_move_encoded=np.array(encoding_move[last_move]).reshape(1,-1)
        predicted_probs = model.predict(last_move_encoded)
        predicted_move_index = np.argmax(predicted_probs)
        predicted_move = ['R', 'P', 'S'][predicted_move_index]
    else:
        predicted_move = 'R' 
    
    return predicted_move
