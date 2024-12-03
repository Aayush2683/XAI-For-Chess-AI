import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from ChessEngine import GameState
from ChessAI import findBestMoveWithoutQueue, scoreBoard

MAX_MOVES_PER_GAME = 120  # Limit the number of moves per game to avoid infinite loops

def generate_training_data(num_games=100):
    training_data = []
    for i in range(num_games):
        print(f"Starting game {i+1}/{num_games}")
        game_state = GameState()
        move_count = 0
        while not game_state.checkmate and not game_state.stalemate and move_count < MAX_MOVES_PER_GAME:
            valid_moves = game_state.getValidMoves()
            if not valid_moves:
                break
            move = findBestMoveWithoutQueue(game_state, valid_moves)
            game_state.makeMove(move)
            features = extract_features(game_state)
            score = scoreBoard(game_state)
            training_data.append((features, score))
            game_state.white_to_move = not game_state.white_to_move
            move_count += 1
            if move_count % 10 == 0:
                print(f"Game {i+1}: Move {move_count} processed")
        print(f"Game {i+1}/{num_games} completed with {move_count} moves.")
        if game_state.checkmate:
            print(f"Game {i+1} ended in checkmate.")
        elif game_state.stalemate:
            print(f"Game {i+1} ended in stalemate.")
        elif move_count >= MAX_MOVES_PER_GAME:
            print(f"Game {i+1} reached the maximum move limit of {MAX_MOVES_PER_GAME}.")
    return training_data

def extract_features(game_state):
    """Extract comprehensive features from the game state."""
    material_advantage = calculate_material_advantage(game_state)
    center_control = calculate_center_control(game_state)
    king_safety = calculate_king_safety(game_state)
    mobility = calculate_mobility(game_state)
    attack_coverage = calculate_attack_coverage(game_state)
    defense_stance = calculate_defense_stance(game_state)
    features = np.array([
        material_advantage, center_control, king_safety,
        mobility, attack_coverage, defense_stance
    ])
    return features

def calculate_material_advantage(game_state):
    """Calculate the material advantage using a detailed evaluation."""
    piece_value = {'p': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9}
    board = game_state.board
    material_advantage = 0
    for r in range(8):
        for c in range(8):
            piece = board[r][c]
            if piece != "--":
                value = piece_value.get(piece[1], 0)
                material_advantage += value if piece[0] == 'w' else -value
    return material_advantage

def calculate_center_control(game_state):
    """Calculate the center control using a detailed evaluation."""
    center_squares = [(3, 3), (3, 4), (4, 3), (4, 4)]
    center_control = 0
    piece_value = {'p': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9}
    for r, c in center_squares:
        if game_state.board[r][c] != "--":
            piece = game_state.board[r][c]
            control_value = piece_value.get(piece[1], 0)
            center_control += control_value if piece[0] == 'w' else -control_value
    # Consider pieces attacking the center squares
    for move in game_state.getAllPossibleMoves():
        if (move.end_row, move.end_col) in center_squares:
            piece = game_state.board[move.start_row][move.start_col]
            control_value = piece_value.get(piece[1], 0)
            center_control += 0.5 * control_value if piece[0] == 'w' else -0.5 * control_value
    return center_control

def calculate_king_safety(game_state):
    """Calculate the king's safety using a detailed evaluation."""
    board = game_state.board
    king_safety = 0
    king_positions = {'wK': None, 'bK': None}
    piece_value = {'p': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9}
    for r in range(8):
        for c in range(8):
            if board[r][c] == 'wK':
                king_positions['wK'] = (r, c)
            elif board[r][c] == 'bK':
                king_positions['bK'] = (r, c)
    for king, pos in king_positions.items():
        if pos:
            r, c = pos
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 8 and 0 <= nc < 8:
                        piece = board[nr][nc]
                        if piece != "--":
                            control_value = piece_value.get(piece[1], 0)
                            king_safety += control_value if piece[0] == king[0] else -control_value
    # Penalize exposed kings (e.g., no pawns in front of the king)
    for king, pos in king_positions.items():
        if pos:
            r, c = pos
            if king == 'wK' and (r > 1 and all(board[r - 1][c + i] == "--" for i in range(-1, 2))):
                king_safety -= 5
            elif king == 'bK' and (r < 6 and all(board[r + 1][c + i] == "--" for i in range(-1, 2))):
                king_safety -= 5
    return king_safety

def calculate_attack_coverage(game_state):
    """Calculate the number of squares attacked by each side with a nuanced approach."""
    board = game_state.board
    attack_coverage = {'w': 0, 'b': 0}
    piece_moves = game_state.getAllPossibleMoves()
    for move in piece_moves:
        piece = board[move.start_row][move.start_col]
        if piece[0] == 'w':
            attack_coverage['w'] += 1
        else:
            attack_coverage['b'] += 1
    # Factor in the number of attacked key squares (center squares, squares around the king)
    key_squares = [(3, 3), (3, 4), (4, 3), (4, 4)]
    for r, c in key_squares:
        if board[r][c] != "--":
            piece = board[r][c]
            if piece[0] == 'w':
                attack_coverage['w'] += 2
            else:
                attack_coverage['b'] += 2
    return attack_coverage['w'] - attack_coverage['b']

def calculate_defense_stance(game_state):
    """Calculate how well each side's pieces are defended using a detailed heuristic."""
    board = game_state.board
    defense_stance = {'w': 0, 'b': 0}
    piece_value = {'p': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 5}  # Adding value for King to ensure it's included
    for r in range(8):
        for c in range(8):
            piece = board[r][c]
            if piece != "--":
                color = piece[0]
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < 8 and 0 <= nc < 8 and board[nr][nc] != "--" and board[nr][nc][0] == color:
                            defense_stance[color] += piece_value[piece[1]]
    return defense_stance['w'] - defense_stance['b']

def calculate_mobility(game_state):
    """Calculate the number of valid moves available for the current player using a detailed heuristic."""
    return len(game_state.getValidMoves())

if __name__ == '__main__':
    print("Starting training data generation...")
    # Generate training data
    training_data = generate_training_data(num_games=100)

    print("Training data generation complete. Training the model now...")
    # Split features and labels
    features = np.array([data[0] for data in training_data])
    scores = np.array([data[1] for data in training_data])

    # Train the model
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(features, scores)

    # Get feature importances
    feature_importances = model.feature_importances_

    # Save the model and feature importances
    with open('chess_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    feature_importance_df = pd.DataFrame({
        'Feature': ["Material Advantage", "Center Control", "King Safety", "Mobility", "Attack Coverage", "Defense Stance"],
        'Importance': feature_importances
    })
    feature_importance_df.to_csv('feature_importances.csv', index=False)

    print("Model training complete. Feature importances saved to 'feature_importances.csv'.")
