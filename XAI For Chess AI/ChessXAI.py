import numpy as np
import shap
import lime.lime_tabular
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import copy
from ChessEngine import GameState, Move

class ChessXAI:
    def __init__(self, game_state):
        if isinstance(game_state, GameState):
            self.game_state = game_state  # Make sure this is a GameState object
        else:
            raise ValueError("The 'game_state' object must be of type 'GameState'.")
        self.model = self.load_model()
        self.feature_importances = self.load_feature_importances()
        self.initial_features = self.extract_features(self.game_state)  # Pass the whole GameState object
        
        # Collect a broader set of training features for LIME
        self.lime_training_data = self.collect_lime_training_data()
        self.explainer_shap = shap.KernelExplainer(self.model.predict, self.initial_features.reshape(1, -1))
        self.explainer_lime = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.lime_training_data,
            feature_names=self.get_feature_names(),
            mode='regression'
        )

    def load_model(self):
        with open('chess_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model

    def load_feature_importances(self):
        feature_importance_df = pd.read_csv('feature_importances.csv')
        return feature_importance_df

    def get_feature_names(self):
        """Names of the features for explanations."""
        return [
            "Material Advantage", "Center Control", "King Safety",
            "Mobility", "Attack Coverage", "Defense Stance"
        ]

    def extract_features(self, game_state):
        """Extract comprehensive features from the game state."""
        material_advantage = self.calculate_material_advantage(game_state)
        center_control = self.calculate_center_control(game_state.board)
        king_safety = self.calculate_king_safety(game_state)
        mobility = self.calculate_mobility(game_state)
        attack_coverage = self.calculate_attack_coverage(game_state)
        defense_stance = self.calculate_defense_stance(game_state)
        features = np.array([
            material_advantage, center_control, king_safety,
            mobility, attack_coverage, defense_stance
        ])
        return features

    def collect_lime_training_data(self):
        """Collect a broad set of training features for LIME."""
        training_features = []
        initial_game_state = copy.deepcopy(self.game_state)  # Make a copy of the initial game state

        for _ in range(100):  # Collect features from 100 random valid game states
            temp_game_state = copy.deepcopy(initial_game_state)  # Start from the initial game state for each iteration
            for _ in range(10):  # Make a sequence of 10 random moves
                temp_game_state = self.random_game_state(temp_game_state)
            features = self.extract_features(temp_game_state)
            training_features.append(features)
        
        return np.array(training_features)

    def random_game_state(self, game_state):
        """Generate a random valid game state for feature extraction."""
        valid_moves = game_state.getValidMoves()
        random_move = np.random.choice(valid_moves)
        game_state.makeMove(random_move)
        return game_state

    def explain_move(self, game_state):
        """Generate and display SHAP and LIME explanations for the current game state."""
        current_features = self.extract_features(game_state).reshape(1, -1)

        # SHAP explanation
        shap_values = None
        try:
            shap_values = self.explainer_shap.shap_values(current_features)
            SHAP_values = shap_values
            SHAP_values_type = type(shap_values)
            SHAP_values_shape = np.shape(shap_values)
            print("\nGenerating SHAP Explanation...")
            print("SHAP values:", SHAP_values)
            print("SHAP values type:", SHAP_values_type)
            print("SHAP values shape:", SHAP_values_shape)

            shap.initjs()
            force_plot = shap.force_plot(
                self.explainer_shap.expected_value,
                shap_values[0].reshape(-1),  # Ensure correct shape for single sample
                feature_names=self.get_feature_names()
            )

            shap_html_path = "shap_plot.html"
            shap.save_html(shap_html_path, force_plot)
            print("SHAP plot saved as an HTML file.")
        except Exception as e:
            print(f"Error in SHAP visualization: {e}")

        # LIME explanation
        lime_image_path = ""
        lime_exp = None
        try:
            print("\nGenerating LIME Explanation...")
            lime_exp = self.explainer_lime.explain_instance(
                current_features.flatten(),
                self.model.predict,
                num_features=6
            )

            # Save detailed LIME plots for all features
            lime_html = lime_exp.as_html()
            with open('lime_explanation.html', 'w') as f:
                f.write(lime_html)
            
            lime_image_path = 'lime_explanation.html'
            print("LIME plot saved as an HTML file.")
        except Exception as e:
            print(f"Error in LIME visualization: {e}")

        # Generate detailed explanations
        chosen_move_explanation, comparison_table_html, best_features_values, comparison_explanation_html = self.explain_chosen_move(game_state, current_features) if shap_values is not None else ("Explanation could not be generated.", "", "", "")
        
        # Feature importance
        feature_importance_html = """
        <h2>Feature Importance</h2>
        <table>
            <tr>
                <th>Feature</th>
                <th>Importance</th>
            </tr>
        """
        for _, row in self.feature_importances.iterrows():
            feature_importance_html += f"<tr><td>{row['Feature']}</td><td>{row['Importance']}</td></tr>"
        feature_importance_html += "</table>"

        # Score calculation formula
        score_formula_html = """
        <h2>Score Calculation Formula</h2>
        <p>The score for each move is calculated using the following formula:</p>
        <p> <b>Score</b> = <br>
        """
        for _, row in self.feature_importances.iterrows():
            score_formula_html += f"  &emsp;&emsp;  {row['Importance']} * {row['Feature']} +<br>"
        score_formula_html = score_formula_html.rstrip("+<br>") + "</p>"

        # Save detailed explanations to a new HTML file
        try:
            detailed_html = f"""
            <html>
            <head>
                <title>Detailed Explanations</title>
                <style>
                    table {{
                        width: 100%;
                        border-collapse: collapse;
                    }}
                    th, td {{
                        border: 1px solid black;
                        padding: 8px;
                        text-align: left;
                    }}
                    th {{
                        background-color: #f2f2f2;
                    }}
                    .highlight {{
                        background-color: #ffeb3b;
                    }}
                </style>
            </head>
            <body>
                <h1>Detailed Explanations</h1>
                <h2>Feature Meanings</h2>
                <p><b>Material Advantage:</b> The value of the pieces on the board, considering captures and losses.</p>
                <p><b>Center Control:</b> Control over the central squares of the board, which is crucial for overall board dominance.</p>
                <p><b>King Safety:</b> The safety of the king, including factors like castling and piece placement around the king.</p>
                <p><b>Mobility:</b> The number of valid moves available, reflecting the flexibility and potential of pieces.</p>
                <p><b>Attack Coverage:</b> The number of squares controlled or attacked, putting pressure on the opponent's pieces.</p>
                <p><b>Defense Stance:</b> The defensive positioning of pieces, protecting against threats and securing key positions.</p>
                {feature_importance_html}
                {score_formula_html}
                <h2>Current Feature Values</h2>
                <p><b>Chosen Move Feature Values:</b> {self.get_current_feature_values(current_features)}</p>
                <p><b>Best Move Feature Values:</b> {best_features_values}</p>
                <h2>Formulas Used</h2>
                <p><b>For Chosen AI move:</b> feature_diff = chosen_move_features - best_features</p>
                <p><b>For Comparison:</b> feature_diff = comparison_move_features - chosen_move_features</p>
                <h2>Chosen Move Explanation</h2>
                <table>
                    <tbody>
                        {chosen_move_explanation}
                    </tbody>
                </table>
                {comparison_table_html}
                <table>
                    <tbody>
                        {comparison_explanation_html}
                    </tbody>
                </table>
                <h2>SHAP Explanation</h2>
                <p><b>SHAP values:</b> {SHAP_values}</p>
                <p><b>SHAP values shape:</b> {SHAP_values_shape}</p>
                <iframe src="shap_plot.html" width="100%" height="600px"></iframe>
                <h2>LIME Explanation</h2>
                <iframe src="{lime_image_path}" width="100%" height="600px"></iframe>
            </body>
            </html>
            """

            with open('detailed_explanation.html', 'w') as f:
                f.write(detailed_html)

            print("Detailed explanation saved to 'detailed_explanation.html'. Open this file in a web browser to view the detailed explanations.")
        except Exception as e:
            print(f"Error saving detailed explanations: {e}")

    def get_current_feature_values(self, current_features):
        """Get the current feature values in a formatted string."""
        feature_values = self.get_feature_names()
        formatted_values = ""
        for i, feature in enumerate(feature_values):
            formatted_values += f"{feature}: {current_features[0][i]:.2f}, "
        return formatted_values.strip(", ")

    def explain_chosen_move(self, game_state, current_features):
        """Generate detailed reasoning for the chosen move."""
        # Temporarily set the turn to black (ChessAI) and get valid moves
        original_turn = game_state.white_to_move
        game_state.white_to_move = False
        valid_moves = game_state.getValidMoves()
        game_state.white_to_move = original_turn  # Restore original turn

        chosen_move = game_state.move_log[-1]  # Get the last move made by the AI

        # Evaluate features for each valid move
        move_evaluations = []
        for move in valid_moves:
            game_state.makeMove(move)
            features = self.extract_features(game_state)
            game_state.undoMove()
            move_evaluations.append((move, features))

        # Calculate the feature impact differences
        chosen_move_features = self.extract_features(game_state)
        explanations = []
        for move, features in move_evaluations:
            diff = chosen_move_features - features
            explanation = f"For move {move.getChessNotation()}, the feature differences were:\n"
            for i, feature_name in enumerate(self.get_feature_names()):
                explanation += f"  - {feature_name}: {diff[i]:.2f}\n"
            explanations.append(explanation)

        # Find the best move according to the model
        move_scores = [self.model.predict(features.reshape(1, -1))[0] for _, features in move_evaluations]
        best_move_index = np.argmax(move_scores)
        best_move, best_features = move_evaluations[best_move_index]

        # Sort moves by score in descending order, excluding the best move
        sorted_move_indices = np.argsort(move_scores)[::-1]
        sorted_move_indices = [i for i in sorted_move_indices if i != best_move_index]

        # Explain why the chosen move was better
        chosen_move_explanation = ""
        for i, feature_name in enumerate(self.get_feature_names()):
            feature_diff = chosen_move_features[i] - best_features[i]
            if feature_diff > 0:
                chosen_move_explanation += f"<tr><td>{feature_name}</td><td>Improved by {feature_diff:.2f}</td></tr>"
            elif feature_diff < 0:
                chosen_move_explanation += f"<tr><td>{feature_name}</td><td>Reduced by {abs(feature_diff):.2f}</td></tr>"
            else:
                chosen_move_explanation += f"<tr><td>{feature_name}</td><td>No impact</td></tr>"

        # Add comparison with other valid moves in descending order of their score
        comparison_explanation = ""
        comparison_explanation_rows = []
        for index in sorted_move_indices:
            move, features = move_evaluations[index]
            move_explanation = [f"If the AI had chosen {move.getChessNotation()}"]
            for i, feature_name in enumerate(self.get_feature_names()):
                feature_diff = features[i] - chosen_move_features[i]
                if feature_diff > 0:
                    move_explanation.append(f" Worse by {feature_diff:.2f}")
                elif feature_diff < 0:
                    move_explanation.append(f" Better by {abs(feature_diff):.2f}")
                else:
                    move_explanation.append(f" No difference")
            comparison_explanation_rows.append(move_explanation)

        # Generate HTML table for comparison
        comparison_table_html = f"""
        <h2>Comparison with Other Valid Moves</h2>
        <table>
            <tr>
                <th>Move</th>
                {''.join(f'<th>{feature}</th>' for feature in self.get_feature_names())}
            </tr>
            <tr class="highlight">
                <td>{chosen_move.getChessNotation()}</td>
                {''.join(f'<td>{chosen_move_features[i]:.2f}</td>' for i in range(len(self.get_feature_names())))}
            </tr>
        """
        for index in sorted_move_indices:
            move, features = move_evaluations[index]
            comparison_table_html += f"""
            <tr>
                <td>{move.getChessNotation()}</td>
                {''.join(f'<td>{features[i]:.2f}</td>' for i in range(len(self.get_feature_names())))}
            </tr>
            """
        comparison_table_html += "</table>"

        # Generate HTML table for chosen move explanation
        chosen_move_table_html = f"""
        <table>
            <thead>
                <tr>
                    <th>Feature</th>
                    <th>Impact</th>
                </tr>
            </thead>
            <tbody>
        """
        for detail in chosen_move_explanation:
            chosen_move_table_html += detail
        chosen_move_table_html += "</tbody></table>"

        # Generate HTML table for comparison explanation
        comparison_explanation_html = f"""
        <h2>Comparison Explanation</h2>
        <table>
            <tr>
                <th>Move</th>
                {''.join(f'<th>{feature}</th>' for feature in self.get_feature_names())}
            </tr>
        """
        # Add chosen move explanation row
        comparison_explanation_html += f"""
            <tr class="highlight">
                <td>{chosen_move.getChessNotation()}</td>
                {''.join(f'<td>{self._format_explanation_value(chosen_move_features[i], best_features[i])}</td>' for i in range(len(self.get_feature_names())))}
            </tr>
        """
        # Add other moves explanations
        for move_explanation in comparison_explanation_rows:
            move = move_explanation[0]
            comparison_explanation_html += f"<tr><td>{move}</td>"
            for feature_explanation in move_explanation[1:]:
                comparison_explanation_html += f"<td>{feature_explanation}</td>"
            comparison_explanation_html += "</tr>"
        comparison_explanation_html += "</table>"

        best_features_values = self.get_current_feature_values(best_features.reshape(1, -1))

        return chosen_move_table_html, comparison_table_html, best_features_values, comparison_explanation_html

    def _format_explanation_value(self, chosen_feature_value, best_feature_value):
        """Format the explanation value for the comparison table."""
        feature_diff = chosen_feature_value - best_feature_value
        if feature_diff > 0:
            return f"Improved by {feature_diff:.2f}"
        elif feature_diff < 0:
            return f"Reduced by {abs(feature_diff):.2f}"
        else:
            return "No impact"

    def calculate_material_advantage(self, game_state):
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

    def calculate_center_control(self, board):
        """Calculate the center control using a detailed evaluation."""
        center_squares = [(3, 3), (3, 4), (4, 3), (4, 4)]
        center_control = 0
        piece_value = {'p': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9}
        for r, c in center_squares:
            if board[r][c] != "--":
                piece = board[r][c]
                control_value = piece_value.get(piece[1], 0)
                center_control += control_value if piece[0] == 'w' else -control_value

        # Consider pieces attacking the center squares
        for move in self.game_state.getAllPossibleMoves():
            if (move.end_row, move.end_col) in center_squares:
                piece = board[move.start_row][move.start_col]
                control_value = piece_value.get(piece[1], 0)
                center_control += 0.5 * control_value if piece[0] == 'w' else -0.5 * control_value
        return center_control

    def calculate_king_safety(self, game_state):
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
                if king == 'wK' and (r > 1 and all(0 <= c + i < 8 and board[r - 1][c + i] == "--" for i in range(-1, 2))):
                    king_safety -= 5
                elif king == 'bK' and (r < 6 and all(0 <= c + i < 8 and board[r + 1][c + i] == "--" for i in range(-1, 2))):
                    king_safety -= 5
        return king_safety

    def calculate_attack_coverage(self, game_state):
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

    def calculate_defense_stance(self, game_state):
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

    def calculate_mobility(self, game_state):
        """Calculate the number of valid moves available for the current player using a detailed heuristic."""
        return len(game_state.getValidMoves())
