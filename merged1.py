# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# <a href="https://colab.research.google.com/github/Setebos12/Tick-Tac-Go/blob/main/merged1.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Boards

# %%
from abc import ABC, abstractmethod


# %%
class AbstractBoard(ABC):
    @abstractmethod
    def get_neighbours(self, position):
        """
        Returns a list of neighboring positions for the given position.
        """
        pass

    @abstractmethod
    def get_neighbour(self, position, direction):
        """
        Returns the neighboring position in the specified direction.
        """
        pass
    @abstractmethod
    def get_value(self, position):
        """
        Returns the value at the given position on the board.
        """
        pass
    @abstractmethod
    def set_value(self, position, value):
        """
        Sets the value at the given position on the board.
        This method can be overridden if the board supports setting values.
        """
        pass


# %%
class GridBoard(AbstractBoard):
    def __init__(self, grid):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def get_neighbours(self, position):
        x, y = position
        neighbours = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if (dx != 0 or dy != 0) and 0 <= x + dx < self.rows and 0 <= y + dy < self.cols:
                    neighbours.append((x + dx, y + dy))
        return neighbours

    def get_neighbour(self, position, direction):
        x, y = position
        dx, dy = direction
        new_x, new_y = x + dx, y + dy
        if 0 <= new_x < self.rows and 0 <= new_y < self.cols:
            return (new_x, new_y)
        return None

    def get_value(self, position):
        x, y = position
        return self.grid[x][y] if 0 <= x < self.rows and 0 <= y < self.cols else None


    def set_value(self, position, value):
        x, y = position
        if self.check_if_position_board(position):
            self.grid[x][y] = value

    def check_if_position_board(self, position):
        x, y = position
        return 0 <= x < self.rows and 0 <= y < self.cols

    def __str__(self):
        for row in self.grid:
            print(" ".join(map(str, row)))
        return ""


# %%
class Board(AbstractBoard):
    def __init__(self, neighbours, board):
        self.neighbours = neighbours
        self.board = board

    def get_neigbours(self, position):
        """
        Returns a list of neighboring positions for the given position.
        """
        return self.neighbours.get(position, [])

    def get_value(self, position):
        """
        Returns the value at the given position on the board.
        """
        return self.board.get(position)

    def set_value(self, position, value):
        """
        Sets the value at the given position on the board.
        """
        self.board[position] = value


# %%
class DirectionBoard(Board):
    def __init__(self, neighbours, board, directions):
        super().__init__(neighbours, board)
        self.directions = directions

    def get_neighbour(self, position, direction):
        """
        Returns the neighboring position in the specified direction.
        """
        if direction in self.directions:
            dx, dy = self.directions[direction]
            new_x, new_y = position[0] + dx, position[1] + dy
            return (new_x, new_y) if (new_x, new_y) in self.board else None
        return None


# %% [markdown]
# # Player

# %%
from types import FunctionType
class Player:
    def __init__(self, name: str, score: int = 0, symbol: str = None, agent : FunctionType = None):
        self.name = name
        self.score = score
        self.symbol = symbol
        self.agent = agent

    def make_move(self, board, position, symbol):
        """
        Make a move on the board at the specified position with the given symbol.
        """
        if board.get_value(position) in [None, '', 0]:
            board.set_value(position, symbol)
            return True
        return False

    def make_move_agent(self, *args):
      """
      Make a move on the board at the specified position with the given symbol.
      """
      return self.agent.choose_move(*args)

    def __str__(self):
        return f"{self.name}"


# %% [markdown]
# # Game

# %%
class Game:
    def __init__(self, board, players = None):
        self.players = players
        if self.players is None:
          self.players = [Player("AI", symbol='X'), Player("Bob", symbol='S')]

        self.board = board
        self.current_player_index = 0

    def switch_player(self):
        """
        Switch to the next player.
        """
        self.current_player_index = (self.current_player_index + 1) % len(self.players)


    def run(self):
        """
        Run the game loop.
        """
        while True:
            current_player = self.players[self.current_player_index]
            print(f"{current_player.name}'s turn")
            print(self.board.grid)
            position = self.get_player_move(current_player)
            symbol = current_player.symbol

            if not current_player.make_move(self.board, position, symbol):
                print("Invalid move. Try again.")
                continue

            if Check_all_positions_directions(self.board, symbol, position):
                print(self.board.grid)
                print(f"{current_player.name} wins!")
                break

            self.switch_player()

    def evaluate(self):
        """
        Evaluate the current game state.
        """
        for player in self.players:
            if Check_all_positions_directions(self.board, player.symbol):
                return player
        return None

    def is_terminal_node(self):
        """
        Check if the game is in a terminal state.
        """
        # The game is terminal if there's a winner or the board is full
        return self.evaluate() is not None or not self.possible_moves()


    def get_player_move(self, player):
        """
        Get the player's move input.
        """
        while True:
            try:
                move = input(f"{player.name}, enter your move (row,col): ")
                row, col = map(int, move.split(','))
                if 0 <= row < len(self.board.grid) and 0 <= col < len(self.board.grid[0]):
                    return (row, col)
                else:
                    print("Move out of bounds. Try again.")
            except ValueError:
                print("Invalid input. Please enter row and column as 'row,col'.")

    def possible_moves(self):
        """
        Get a list of possible moves.
        """
        moves = []
        for i in range(len(self.board.grid)):
            for j in range(len(self.board.grid[0])):
                # Check for empty strings, None, or 0 as empty
                if self.board.get_value((i, j)) in [None, '', 0]:
                    moves.append((i, j))
        return moves

    def make_move(self, move):
        """
        Make a move on a *copy* of the board and return a *new* Game object.
        """
        self.board.set_value(move, self.players[self.current_player_index].symbol)
        self.switch_player()


    def deep_copy(self):
        """
        Create a deep copy of the game.
        """
        new_players = [Player(player.name, player.score, player.symbol) for player in self.players]
        new_board = GridBoard([row[:] for row in self.board.grid]) # Correctly copy the grid
        new_game = Game(new_board)
        new_game.players = new_players
        new_game.current_player_index = self.current_player_index
        return new_game

    def current_player(self):
      return self.players[self.current_player_index]

    def info(self):
      info = {"players" : len(self.players),
              "turn" : self.current_player_index,
              "board": self.board.grid}
      return info


# %% [markdown]
# # Rules

# %%
def All_same_cryteria(nums, symbol):
    for num in nums:
        if num != symbol:
            return False
    return True


# %%
def Check_Connection(nums, symbol, position, direction, how_many = 3):
    """
    Check if there is a connection of the same symbol in the specified direction.
    """
    count = 0
    for i in range(how_many):
        if  nums.get_value(position) == symbol:
            count += 1
            position = (position[0] + direction[0], position[1] + direction[1])
        else:
            break
    return count == how_many


# %%
def Check_all_directions(nums, symbol, position, directions=None, how_many = 3):
    """
    Check if there is a connection in any of the specified directions.
    """
    if directions is None:
        directions = [(-1, -1), (-1, 0), (-1, 1),
                    (0, -1),           (0, 1),
                    (1, -1),  (1, 0),  (1, 1)]
    for direction in directions:
        if Check_Connection(nums, symbol, position, direction, how_many):
            return True
    return False


# %%
def Check_all_positions_directions(nums, symbol, directions=None, how_many = 3):
    """
    Check if there is a connection in any of the specified directions.
    """
    for i in range(len(nums.grid)):
        for j in range(len(nums.grid[0])):
            if Check_all_directions(nums, symbol, (i, j)):
                return True
    return False


# %% [markdown]
# # Node
#
# representing each state of the game

# %%
class Node:
    def __init__(self, game, parent=None, move=None):
        self.game = game
        self.parent = parent
        self.move = move
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def add_children(self):
        for move in self.game.possible_moves():
            new_game = self.game.deep_copy()
            new_game.make_move(move)
            child = Node(new_game, self, move)
            self.add_child(child)

    def is_terminal_node(self):
        return self.game.is_terminal_node()

    def evaluate(self, player = 0):
        evaluation = self.game.evaluate()
        if isinstance(evaluation, Player):
            evaluation1 = 1 if evaluation.symbol == 'X' else -1
            if player != 0:
                evaluation1 = -evaluation1
            return evaluation1
        return 0

    def state_key(self):
        return str(self.game.info())

    def __str__(self) -> str:
        return str(self.game.info())


    def get_parent_move(self):
        return self.move ,self.parent


# %% [markdown]
# # min max algorythms
#
# * normal (brute force)
# * memotyzation all states
# * just the best move
#
# TO_DO for each postion you get sorted by evaluaion moves
#
#
# Jak sprawdzić czy mam już zrobione? i mogę isc dalej.
#  - napisać testy czy dla kazdej pozycji jest ruch w wymaganym czasie
#  - napisać testy które sprawdzą poprawność algorytmu

# %%
def min_max(node, depth, maximizing_player):
    if depth == 0 or node.is_terminal_node():
        return node.evaluate()

    if maximizing_player:
        max_eval = -float('inf')
        node.add_children()
        for child in node.children:
            eval = min_max(child, depth - 1, False)
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float('inf')
        node.add_children()
        for child in node.children:
            eval = min_max(child, depth - 1, True)
            min_eval = min(min_eval, eval)
        return min_eval


# %%
def min_max_remember(node, depth, maximizing_player, memo):
    if depth == 0 or node.is_terminal_node():
        return node.evaluate(), None


    best_move = None

    if maximizing_player:
        max_eval = -float('inf')
        node.add_children()
        for child in node.children:
            eval, _ = min_max_remember(child, depth - 1, False, memo)
            if eval > max_eval:
                max_eval = eval
                best_move = child.move
            add_to_memo(memo,node, child.move, eval, maximizing_player) # Moved inside the loop
        return max_eval, best_move

    else:
        min_eval = float('inf')
        node.add_children()
        for child in node.children:
            eval, _ = min_max_remember(child, depth - 1, True, memo)
            if eval < min_eval:
                min_eval = eval
                best_move = child.move
            add_to_memo(memo,node, child.move, eval, maximizing_player) # Moved inside the loop
        return min_eval, best_move


def add_to_memo(memo,node, move, eval, maximizing_player):
  if str(node) not in memo:
    memo[str(node)] = [(eval, move)]
  else:
    if (eval, move) not in memo[str(node)]:
      memo[str(node)].append((eval, move))
      if maximizing_player:
        memo[str(node)].sort(reverse=True)
      else:
        memo[str(node)].sort()


# %%
grid = [
      ['', '', ''],
      ['', '', ''],
      ['', '', '']
  ]
board = GridBoard(grid)
game = Game(board)
node = Node(game)
memo = {}
min_max_remember(node, 10000000, 1, memo)


# %%
from random import choice
class memoagent:
  def __init__(self, memo):
    self.memo = memo
    self.is_learning = False

  def choose_move(self,node, *args):
    moves = memo[str(node)]
    best_moves = [move for eval, move in moves if eval == moves[0][0]]
    move = choice(best_moves)
    return move

  def choose_action(self, node, *args):
    moves = memo[str(node)]
    best_moves = [move for eval, move in moves if eval == moves[0][0]]
    move = choice(best_moves)
    return move



class memoagent1:
  def __init__(self, memo):
    self.memo = memo

  def choose_move(self, game, *args):
    node = Node(game)
    moves = memo[str(node)]
    return moves[0][0]

class random_player:
  def __init__(self):
    self.is_learning = False


  def choose_move(self, node, *args):
    return choice(node.game.possible_moves())


  def choose_action(self, node, *args):
    return choice(node.game.possible_moves())


# %%
def find_best_move(node):
    best_eval = -float('inf')
    best_move = None
    node.add_children()
    ans = []
    for child in node.children:
        eval = min_max(child, 100, False)
        if eval > best_eval:
            best_eval = eval
            best_move = child.move

        ans.append((child.move, eval))
    return ans


# %% [markdown]
# # Veryfictaion
#
#
#

# %%
class Veryfication:
    def __init__(self, players):
        self.players = players
        self.results = dict()
        self.create_dict()

    def create_dict(self):
        endings = ["win", "lose", "draw"]
        for player in self.players:
            self.results[player.name] = {end: 0 for end in endings}

    def verify(self, episodes):
        for episode in range(episodes):
            game = self.setup_game()
            self.play_game(game)
            self.record_result(game)

    def setup_game(self):
        grid = [['', '', ''], ['', '', ''], ['', '', '']]
        board = GridBoard(grid)
        return Game(board, self.players)

    def play_game(self, game):
        current_player = None
        while not game.is_terminal_node():
            current_player = game.players[game.current_player_index]
            move = current_player.make_move_agent(Node(game))
            game.make_move(move)


    def record_result(self, game):
        who_won = game.evaluate()
        if isinstance(who_won, Player):
            self.results[who_won.name]["win"] += 1
            for player in self.players:
                if player != who_won:
                    self.results[player.name]["lose"] += 1
        else:
            for player in self.players:
                self.results[player.name]["draw"] += 1


# %%
class VeryficationV2(Veryfication):
  def __init__(self, players):
    super().__init__(players)
    self.game_history = None


  def verify(self, episodes):
    for episode in range(episodes):
      game = self.setup_game()
      self.game_history = GameHistory(game)
      self.play_game(game)
      self.record_result(game)

  def play_game(self, game):
      while not game.is_terminal_node():
          current_player = game.players[game.current_player_index]
          node = self.game_history.get_current()
          move = current_player.make_move_agent(node, True)
          game.make_move(move)
          self.game_history.save_state(game, move)

      self.last_round_learn(self.game_history.get_current())

  def last_round_learn(self, node):
    for player in self.players:
      if player.agent.is_learning:
        node.game.current_player_index = self.players.index(player)
        player.agent.learn_in_game_from_previous_moves(node)


# %%
class GameHistory:
  def __init__(self, game):
    self.begin_node = Node(game)
    self.current_node = self.begin_node

  def save_state(self, game, move):
    new_game = game.deep_copy()
    node = Node(new_game, parent=self.current_node, move=move)
    self.current_node.add_child(node)
    self.current_node = node


  def get_current(self):
    return self.current_node



# %%
player1 = Player("Ala", 0 , "X", memoagent(memo))
player2 = Player("Bob", 0 , "S", random_player())
players = [player1, player2]
veryfiy = VeryficationV2(players)
veryfiy.verify(1)
veryfiy.results

# %%
gm = veryfiy.game_history


def retrive_game(game_history):
  ans = []
  current_node = game_history.current_node
  while current_node.parent is not None:
    ans.append(current_node.game.info())
    print(current_node.game.info())
    current_node = current_node.parent
  return ans.reverse()


ans  = retrive_game(gm)

# %%
gm = veryfiy.game_history.begin_node.game.info()
gm

# %% [markdown]
# # Q algorythm
# :
# TO_DO 🇹:
#
#
# 1.   move Q algorytm to class or funtion 🚂 🚆
# 2.   Test algorytms with min_max and evalution functio 🎮
#
# SUPER TO_DO
#
#
#
# 1.   another types of Q algorytm  
#
#
#
#
#
#

# %%
import random

class Qagent:
  def __init__(self, Q):
    self.Q = Q

  def choose_move(self, node, *args):
    state = node.state_key()
    possible_moves = node.game.possible_moves()
    q_vals = [self.Q.get((state, a), 0) for a in possible_moves]
    max_q = max(q_vals)
    best_moves = [a for a in possible_moves if self.Q.get((state, a), 0) == max_q]
    action = random.choice(best_moves)
    return action


# %%
class QAgentTrain(Qagent):
    def __init__(self):
        self.Q = {}
        self.last_action = None
        self.is_learning = True

    def step(self, node, action):
        if action not in node.game.possible_moves():

            return node, -10

        player = node.game.current_player_index
        new_game = node.game.deep_copy()
        new_game.make_move(action)
        new_node = Node(new_game, parent=node, move=action)

        reward = new_node.evaluate(player)
        return new_node, reward

    def choose_action(self, node, epsilon):
        state = node.state_key()
        possible_moves = node.game.possible_moves()
        self.random_action = False
        if random.random() < epsilon:
            self.random_action = True
            return random.choice(possible_moves)

        q_vals = [self.Q.get((state, a), 0) for a in possible_moves]
        max_q = max(q_vals)
        best_moves = [a for a in possible_moves if self.Q.get((state, a), 0) == max_q]
        return random.choice(best_moves)

    def update_q_value(self, state, action, reward, next_state, next_possible_moves, gamma, alpha, Q = None):
        future_q = max([self.Q.get((next_state, a), 0) for a in next_possible_moves], default=0)
        old_q = self.Q.get((state, action), 0)
        self.Q[(state, action)] = old_q + alpha * (reward + gamma * future_q - old_q)

    def initialize_game(self):
        grid = [['', '', ''], ['', '', ''], ['', '', '']]
        board = GridBoard(grid)
        game = Game(board)
        return Node(game)

    def train_episode(self, rival, epsilon, gamma, alpha):
        node = self.initialize_game()

        while not node.is_terminal_node():
            self.train_step(node, rival, epsilon, gamma, alpha)
            if node.is_terminal_node():
                break
            node = self.next_node  # Store the new state from train_step for next loop


    def train_step(self, node, rival, epsilon, gamma, alpha):
        state = node.state_key()
        action = self.choose_action(node, epsilon)
        next_node, reward1 = self.step(node, action)


        if rival.is_learning:
           rival.learn_in_game_from_previous_moves(next_node, gamma, alpha)

        if next_node.is_terminal_node():
            self.learn_in_game_end(next_node, alpha)

            # self.Q[(state, action)] = self.Q.get((state, action), 0) + alpha * (reward1 - self.Q.get((state, action), 0))
            self.next_node = next_node
            return


        reward = 0

        minimax_action = rival.choose_action(next_node, 0)
        next_node2, reward = self.step(next_node, minimax_action)



        # next_state = next_node.state_key()
        # next_possible_moves = next_node.game.possible_moves()
        # total_reward = reward1 + reward
        # self.rewards.append(reward)

        # self.update_q_value(state, action, total_reward, next_state, next_possible_moves, gamma, alpha)

        self.learn_in_game_from_previous_moves(next_node2, gamma, alpha)


        if rival.is_learning and next_node2.is_terminal_node():
            # state = next_node.state_key()
            rival.learn_in_game_end(next_node2, alpha)

            # rival.Q[(state, minimax_action)] = rival.Q.get((state, minimax_action), 0) + alpha * (reward - rival.Q.get((state, minimax_action), 0))

        self.next_node = next_node2

    def learn_in_game_from_previous_moves(self, node : Node, gamma, alpha):
        # total_reward = 0
        # current_node = node
        # player = node.game.current_player()
        # action = node.move
        # while current_node.parent is not None:
        #   action, parent_node = current_node.get_parent_move()
        #   p_player = parent_node.game.current_player()
        #   reward = current_node.evaluate(p_player)
        #   if p_player == player:
        #     total_reward *= -1
        #     total_reward += reward
        #     break
        #   else:
        #     total_reward += current_node.evaluate(p_player)

        #   current_node = parent_node

        opponent_previous_move = node.parent
        my_previous_move = node.parent.parent
        if my_previous_move is None:
          return


        my_state = my_previous_move.state_key()
        my_action = opponent_previous_move.move
        opponent_state = opponent_previous_move.state_key()
        opponent_action = node.move

        _, my_reward = self.step(my_previous_move, my_action)
        _, opponent_reward = self.step(opponent_previous_move, opponent_action)
        total_reward = my_reward - opponent_reward

        self.update_q_value(my_state, my_action, total_reward, node.state_key(), node.game.possible_moves(), gamma, alpha)


    def learn_in_game_end(self, node: Node, alpha):
       if node.is_terminal_node() is not True:
        return

       my_previous_move = node.parent
       my_state = my_previous_move.state_key()
       my_action = node.move
       _, my_reward = self.step(my_previous_move, my_action)


       self.Q[(my_state, my_action)] = self.Q.get((my_state, my_action), 0) + alpha * (my_reward - self.Q.get((my_state, my_action), 0))


    def train(self, rival=None, episodes=4000, epsilon=0.1, gamma=0.95, alpha=0.1):
        self.rewards = []
        for episode in range(episodes):
            self.train_episode(rival, epsilon, gamma, alpha)


    # def choose_move(self,game, node : Node, learn = False):
    #    if str(game.info()) != str(node.game.info()):
    #     print('hahahah')
    #    move = self.choose_move(node.game)

    #    if learn:
    #     self.learn_move(node)
    #    return move


# %% [markdown]
# # QV2
#
#
#

# %%
class QAgentTrainV2(Qagent):
    def __init__(self, epsilon=0.1, gamma=0.95, alpha=0.1):
        self.Q = {}
        self.last_action = None
        self.is_learning = True

        # epsl
        self.epsl = (0, (0, 1))

        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.rewards = []
        self.epsch = epsilonchanger()

    def step(self, node, action):
        if action not in node.game.possible_moves():

            return node, -10

        player = node.game.current_player_index
        new_game = node.game.deep_copy()
        new_game.make_move(action)
        new_node = Node(new_game, parent=node, move=action)

        reward = new_node.evaluate(player)
        return new_node, reward

    def choose_action(self, node):
        state = node.state_key()
        possible_moves = node.game.possible_moves()
        self.random_action = False
        if random.random() < self.epsilon:
            self.random_action = True
            return random.choice(possible_moves)

        q_vals = [self.Q.get((state, a), 0) for a in possible_moves]
        max_q = max(q_vals)
        best_moves = [a for a in possible_moves if self.Q.get((state, a), 0) == max_q]
        return random.choice(best_moves)

    def update_q_value(self, state, action, reward, next_state, next_possible_moves):
        future_q = max([self.Q.get((next_state, a), 0) for a in next_possible_moves], default=0)
        old_q = self.Q.get((state, action), 0)
        self.Q[(state, action)] = old_q + self.alpha * (reward + self.gamma * future_q - old_q)

    def initialize_game(self):
        grid = [['', '', ''], ['', '', ''], ['', '', '']]
        board = GridBoard(grid)
        game = Game(board)
        return Node(game)

    def train_episode(self, rival):
        node = self.initialize_game()

        while not node.is_terminal_node():
            self.train_step(node, rival)
            if node.is_terminal_node():
                break
            node = self.next_node


    def train_step(self, node, rival):
        state = node.state_key()
        action = self.choose_action(node)
        next_node, reward1 = self.step(node, action)


        if rival.is_learning:
           rival.learn_in_game_from_previous_moves(next_node)

        if next_node.is_terminal_node():
            self.learn_in_game_end(next_node)

            self.next_node = next_node
            return


        reward = 0

        minimax_action = rival.choose_action(next_node) # when choose_action agent is worse when choose_move
        next_node2, reward = self.step(next_node, minimax_action)



        self.learn_in_game_from_previous_moves(next_node2)


        if rival.is_learning and next_node2.is_terminal_node():
            rival.learn_in_game_end(next_node2)


        self.next_node = next_node2

    def learn_in_game_from_previous_moves(self, node : Node):
        if node is None:
          return
        total_reward = 0
        current_node = node
        player = node.game.current_player_index
        action = node.move
        parent_node = None
        while current_node.parent is not None:
          action, parent_node = current_node.get_parent_move()
          if parent_node is None:
            return
          p_player = parent_node.game.current_player_index
          reward = current_node.evaluate(p_player)
          if p_player == player:
            total_reward *= -1
            total_reward += reward
            break
          else:
            total_reward += current_node.evaluate(p_player)

          current_node = parent_node
        if parent_node is None:
            return

        my_state = parent_node.state_key()
        my_action = action

        # opponent_previous_move = node.parent
        # if opponent_previous_move is None:

        #   return
        # my_previous_move = node.parent.parent
        # if my_previous_move is None:
        #   return

        # my_state = my_previous_move.state_key()
        # my_action = opponent_previous_move.move
        # opponent_state = opponent_previous_move.state_key()
        # opponent_action = node.move

        # _, my_reward = self.step(my_previous_move, my_action)
        # _, opponent_reward = self.step(opponent_previous_move, opponent_action)
        # total_reward = my_reward - opponent_reward

        if node.is_terminal_node():
          self.Q[(my_state, my_action)] = self.Q.get((my_state, my_action), 0) + self.alpha * (total_reward - self.Q.get((my_state, my_action), 0))
        else:
          self.update_q_value(my_state, my_action, total_reward, node.state_key(), node.game.possible_moves())

    def learn_move(self, node : Node):
      if node.is_terminal_node():
        self.learn_in_game_end(node)
      else:
        self.learn_in_game_from_previous_moves(node)


    def learn_in_game_end(self, node: Node):
       if node.is_terminal_node() is not True:
        return

       my_previous_move = node.parent
       my_state = my_previous_move.state_key()
       my_action = node.move
       _, my_reward = self.step(my_previous_move, my_action)


       self.Q[(my_state, my_action)] = self.Q.get((my_state, my_action), 0) + self.alpha * (my_reward - self.Q.get((my_state, my_action), 0))


    def train(self, rival=None, episodes=4000):
        self.rewards = []
        for episode in range(episodes):
            self.train_episode(rival)

            self.epsilon = self.epsch.add()
            rival.epsilon = self.epsch.next_change_table()

    def choose_move(self, node, *args):
       move = super().choose_move(node, *args)
       if args[0] is True:
        self.learn_move(node)
       return move

    # def choose_move(self, node, learn = False):
    #    su
    #    move = self.choose_move(node, Fa)
    #    if learn:
    #     self.learn_move(node)
    #    return move


# %%
class epsilonchanger:
  def __init__(self, episodes = 1, change_table = [0.1, 0]) -> None:
     self.episodes = episodes
     self.current_episode = 0
     self.change_table = change_table
     self.current_index = 0

  def add(self):
    self.current_episode += 1
    if self.current_episode == self.episodes:
      self.current_episode = 0
      self.current_index += 1
      if self.current_index == len(self.change_table):
        self.current_index = 0

    return self.change_table[self.current_index]

  def next_change_table(self):
    return self.change_table[(self.current_index + 1) % len(self.change_table)]


# %% [markdown]
# Agenty które obserwują losowe ruchy przeciwnika i same nie wykonują losowych ruchów są lepsze -> zabawa z epsilonem\

# %% [markdown]
# # Nowa sekcja

# %%
agent = QAgentTrainV2()
agent1 = QAgentTrainV2()

agent.train(agent1, 4000)

# %% [markdown]
# Agent lepiej się uczy gdy
#

# %%
grid = [
      ['', '', ''],
      ['', '', ''],
      ['', '', '']
  ]
board = GridBoard(grid)
game = Game(board)

player1 = Player("Ala", 0 , "X", agent)
player2 = Player("Bob", 0 , "S", memoagent(memo))
players = [player1, player2]
veryfiy = VeryficationV2(players)
veryfiy.verify(5000)
veryfiy.results

# %%
grid = [
      ['', '', ''],
      ['', '', ''],
      ['', '', '']
  ]
board = GridBoard(grid)
game = Game(board)

player1 = Player("Ala", 0 , "X", memoagent(memo))
player2 = Player("Bob", 0 , "S", agent1)
players = [player1, player2]
veryfiy = VeryficationV2(players)
veryfiy.verify(5000)
veryfiy.results

# %%
grid = [
      ['', '', ''],
      ['', '', ''],
      ['', '', '']
  ]
board = GridBoard(grid)
game = Game(board)

player1 = Player("Ala", 0 , "X", random_player())
player2 = Player("Bob", 0 , "S", agent1)
players = [player1, player2]
veryfiy = VeryficationV2(players)
veryfiy.verify(5000)
veryfiy.results

# %%
grid = [
      ['', '', ''],
      ['', '', ''],
      ['', '', '']
  ]
board = GridBoard(grid)
game = Game(board)

player1 = Player("Ala", 0 , "X", agent)
player2 = Player("Bob", 0 , "S", random_player())
players = [player1, player2]
veryfiy = VeryficationV2(players)
veryfiy.verify(5000)
veryfiy.results

# %%
grid = [
      ['', '', ''],
      ['', '', ''],
      ['', '', '']
  ]
board = GridBoard(grid)
game = Game(board)

player1 = Player("Ala", 0 , "X", agent)
player2 = Player("Bob", 0 , "S", agent1)
players = [player1, player2]
veryfiy = VeryficationV2(players)
veryfiy.verify(5000)
veryfiy.results

# %%
grid = [
      ['', '', ''],
      ['', '', ''],
      ['', '', '']
  ]
board = GridBoard(grid)
game = Game(board)

player1 = Player("Ala", 0 , "X", memoagent(memo))
player2 = Player("Bob", 0 , "S", random_player())
players = [player1, player2]
veryfiy = VeryficationV2(players)
veryfiy.verify(5000)
veryfiy.results

# %%
# agent.train(agent1, 8000)
# agent.train(random_player(), 4000)
# agent.train(memoagent(memo), 4000)

# %% [markdown]
# # Turnament

# %% [markdown]
# # Nowa sekcja

# %%
import unittest




class TestTicTacToeAgent(unittest.TestCase):

    def test_win_in_row(self):
        board = ['X', 'X', ' ',  # <- agent powinien zagrać na 2
                 ' ', ' ', ' ',
                 ' ', ' ', ' ']
        move = agent_move(board, 'X')
        self.assertEqual(move, 2)

    def test_win_in_column(self):
        board = ['O', ' ', ' ',
                 'O', ' ', ' ',
                 ' ', ' ', ' ']  # <- agent 'O' powinien zagrać na 6
        move = agent_move(board, 'O')
        self.assertEqual(move, 6)

    def test_win_in_diagonal(self):
        board = ['X', ' ', ' ',
                 ' ', 'X', ' ',
                 ' ', ' ', ' ']  # <- agent powinien zagrać na 8
        move = agent_move(board, 'X')
        self.assertEqual(move, 8)

    def test_no_win_play_first_free(self):
        board = ['X', 'O', 'X',
                 'O', 'X', 'O',
                 ' ', ' ', ' ']  # agent nie może wygrać, powinien zagrać na 6
        move = agent_move(board, 'X')
        self.assertEqual(move, 6)

    def test_block_opponent_win(self):
        board = ['O', 'O', ' ',
                 ' ', 'X', ' ',
                 'X', ' ', ' ']  # <- agent 'X' powinien zablokować na 2
        move = agent_move(board, 'X')
        self.assertEqual(move, 2)

    def test_block_column(self):
        board = [' ', 'O', 'X',
                 ' ', 'O', ' ',
                 ' ', ' ', ' ']  # <- agent 'X' powinien zablokować na 7
        move = agent_move(board, 'X')
        self.assertEqual(move, 7)

    def test_fork_move(self):
        board = ['X', ' ', ' ',
                 ' ', 'O', ' ',
                 ' ', ' ', 'X']  # <- agent 'X' może utworzyć forka na 1
        move = agent_move(board, 'X')
        self.assertEqual(move, 1)

# Uruchomienie testów w Colabie
unittest.main(argv=[''], verbosity=2, exit=False)


# %%
grid = [
      ['', '', ''],
      ['', '', ''],
      ['', '', '']
  ]



board = GridBoard(grid)
game = Game(board)

mem =  memoagent(memo)
mem.choose_move(Node(game))

# %%
mem =  memoagent(memo)
random_player()

# %%
agents = {
    "memo": memoagent(memo=memo),
    "random": random_player(),
    "Qagent": agent
}

for name, agent in agents.items():
    try:
        grid = [['X', 'X', ''], ['', '', ''], ['', 'S', 'S']]
        board = GridBoard(grid)
        game = Game(board)
        node = Node(game)

        move = agent.choose_move(node)
        assert move == (0, 2), f"{name} failed: expected (0,2), got {move}"
        print(f"✅ {name} passed")

    except AssertionError as e:
        print(f"❌ Test failed for {name}: {e}")
