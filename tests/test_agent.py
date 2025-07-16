from merged1 import random_player, GridBoard, Game, Node
import pytest


from merged1 import random_player, GridBoard, Game, Node
import pytest


@pytest.mark.parametrize("agent_factory", [
    lambda: random_player(),
])
def test_win_move(agent_factory):
    """Test: agent powinien wygrać (X, X, '')"""
    agent = agent_factory()
    grid = [['X', 'X', ''], ['', '', ''], ['', 'S', 'S']]
    board = GridBoard(grid)
    game = Game(board)
    node = Node(game)

    move = agent.choose_move(node)
    assert move == (0, 2)


@pytest.mark.parametrize("agent_factory", [
    lambda: random_player(),
])

def test_block_opponent(agent_factory):
    """Test: agent powinien zablokować przeciwnika (O, O, '')"""
    agent = agent_factory()
    grid = [['S', 'S', ''], ['X', 'X', ''], ['', '', '']]
    board = GridBoard(grid)
    game = Game(board)
    node = Node(game)

    move = agent.choose_move(node)
    assert move == (0, 2)  # blokuje 'O', które ma 2 w rzędzie


@pytest.mark.parametrize("agent_factory", [
    lambda: random_player(),
])
def test_double_attack(agent_factory):
    """
    Test: agent powinien wykonać podwójny atak (fork)
    Przykład: agent (X) ma dwa możliwe wygrane przy jednym ruchu.
    """
    agent = agent_factory()
    grid = [['X', '', ''],
            ['', 'S', ''],
            ['', '', 'X']]
    board = GridBoard(grid)
    game = Game(board)
    node = Node(game)

    move = agent.choose_move(node)
    assert move in [(0, 2), (2, 0)]


@pytest.mark.parametrize("agent_factory", [
    lambda: random_player(),
])

def test_block_vertical(agent_factory):
    """Agent powinien zablokować przeciwnika na linii pionowej."""
    agent = agent_factory()
    grid = [
        ['O', 'X', ''],
        ['O', '', ''],
        ['', '', 'X'],
    ]
    board = GridBoard(grid)
    game = Game(board)
    node = Node(game)

    move = agent.choose_move(node)
    assert move == (2, 0)

@pytest.mark.parametrize("agent_factory", [
    lambda: random_player(),
])


def test_fork_creation(agent_factory):
    """
    Agent powinien wykonać ruch tworzący "fork" (podwójny atak),
    czyli sytuację, w której ma dwa sposoby na wygraną w następnym ruchu.
    """
    agent = agent_factory()
    grid = [
        ['X', '', ''],
        ['', 'O', ''],
        ['', '', 'X'],
    ]
    board = GridBoard(grid)
    game = Game(board)
    node = Node(game)

    move = agent.choose_move(node)
    # Ruchy, które tworzą fork
    assert move in [(0, 2), (2, 0)]



@pytest.mark.parametrize("agent_factory", [
    lambda: random_player(),
])
def test_take_center(agent_factory):
    """Agent powinien zająć środek planszy, jeśli jest wolny."""
    agent = agent_factory()
    grid = [
        ['X', '', ''],
        ['', '', ''],
        ['', '', 'O'],
    ]
    board = GridBoard(grid)
    game = Game(board)
    node = Node(game)

    move = agent.choose_move(node)
    assert move == (1, 1)


def test_first_move(agent_factory):
    """Agent wykonuje pierwszy ruch na pustej planszy."""
    agent = agent_factory()
    grid = [
        ['', '', ''],
        ['', '', ''],
        ['', '', ''],
    ]
    board = GridBoard(grid)
    game = Game(board)
    node = Node(game)

    move = agent.choose_move(node)
    # Dobry ruch na start: środek lub rogi
    assert move in [(1, 1), (0, 0), (0, 2), (2, 0), (2, 2)]
