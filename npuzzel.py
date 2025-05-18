import tkinter as tk
from tkinter import messagebox, ttk
import random
import heapq
import time


# --- PUZZLE LOGIC + SOLVER ---
class TilePuzzle:
    def __init__(self, puzzle):
        self.puzzle = puzzle
        self.size = len(puzzle)
        self.zero = self.findZero()
        self.moves = ['L', 'R', 'U', 'D']
        self.start_time = 0
        self.solve_time = 0
        self.steps = 0
        self.nodes_expanded = 0
        self.path_length = 0

    def findZero(self):
        for i in range(self.size):
            for j in range(self.size):
                if self.puzzle[i][j] == 0:
                    return (i, j)
        return None

    def checkPuzzle(self):
        expected = list(range(1, self.size * self.size)) + [0]
        flat_puzzle = [val for row in self.puzzle for val in row]
        return flat_puzzle == expected

    def clone(self):
        return TilePuzzle([row[:] for row in self.puzzle])

    def doMove(self, m):
        x, y = self.zero
        moved = False

        # Fix the directions - in a sliding puzzle, you move the tile TO the empty space
        if m == 'L' and y > 0:  # Move tile to the left (empty space moves right)
            self.puzzle[x][y], self.puzzle[x][y-1] = self.puzzle[x][y-1], self.puzzle[x][y]
            self.zero = (x, y-1)
            moved = True
        elif m == 'R' and y < self.size-1:  # Move tile to the right (empty space moves left)
            self.puzzle[x][y], self.puzzle[x][y+1] = self.puzzle[x][y+1], self.puzzle[x][y]
            self.zero = (x, y+1)
            moved = True
        elif m == 'U' and x > 0:  # Move tile up (empty space moves down)
            self.puzzle[x][y], self.puzzle[x-1][y] = self.puzzle[x-1][y], self.puzzle[x][y]
            self.zero = (x-1, y)
            moved = True
        elif m == 'D' and x < self.size-1:  # Move tile down (empty space moves up)
            self.puzzle[x][y], self.puzzle[x+1][y] = self.puzzle[x+1][y], self.puzzle[x][y]
            self.zero = (x+1, y)
            moved = True

        if moved:
            self.steps += 1
        return moved

    def getState(self):
        return tuple(val for row in self.puzzle for val in row)

    def manhattan(self):
        # Manhattan distance heuristic
        dist = 0
        for i in range(self.size):
            for j in range(self.size):
                val = self.puzzle[i][j]
                if val == 0:
                    continue
                tx, ty = (val - 1) // self.size, (val - 1) % self.size
                dist += abs(i - tx) + abs(j - ty)
        return dist

    def misplaced_tiles(self):
        # Misplaced tiles heuristic (Hamming distance)
        count = 0
        for i in range(self.size):
            for j in range(self.size):
                val = self.puzzle[i][j]
                if val == 0:
                    continue
                correct_val = i * self.size + j + 1
                if val != correct_val:
                    count += 1
        return count

    def get_neighbors(self):
        neighbors = []
        for move in self.moves:
            new_puzzle = self.clone()
            if new_puzzle.doMove(move):  # Only add if move was valid
                neighbors.append((move, new_puzzle))
        return neighbors

    def solve(self, heuristic='manhattan'):
        self.start_time = time.time()
        self.steps = 0
        self.nodes_expanded = 0
        start = self.clone()
        goal = tuple(list(range(1, self.size * self.size)) + [0])

        # Select heuristic function
        h_func = self.manhattan if heuristic == 'manhattan' else self.misplaced_tiles

        heap = [(h_func(), 0, start.getState(), [], start)]
        visited = set()

        while heap:
            f, g, state, path, puzzle = heapq.heappop(heap)
            if state in visited:
                continue
            visited.add(state)
            self.nodes_expanded += 1

            if state == goal:
                self.solve_time = time.time() - self.start_time
                self.path_length = len(path)
                return path

            for move, neighbor in puzzle.get_neighbors():
                n_state = neighbor.getState()
                if n_state not in visited:
                    h = h_func() if heuristic == 'manhattan' else neighbor.misplaced_tiles()
                    heapq.heappush(heap, (
                        g + 1 + h,  # f = g + h
                        g + 1,
                        n_state,
                        path + [move],
                        neighbor
                    ))

        self.solve_time = time.time() - self.start_time
        return None


# --- GUI CLASS ---
class PuzzleGUI:
    def __init__(self, root):
        self.root = root
        self.size = 3  # Default size
        self.puzzle = None
        self.buttons = []
        self.root.configure(bg='#222')
        self.time_label = None
        self.steps_label = None
        self.nodes_label = None
        self.path_length_label = None
        self.heuristic_var = tk.StringVar(value='manhattan')
        self.size_var = tk.IntVar(value=3)

        self.create_config_frame()
        self.init_puzzle()
        self.create_widgets()
        self.root.bind('<Key>', self.handle_keypress)
        self.update_metrics()

    def create_config_frame(self):
        # Frame for configuration options
        config_frame = tk.Frame(self.root, bg='#222')
        config_frame.pack(pady=10)

        # Size selection
        size_label = tk.Label(config_frame, text="Puzzle Size:", bg='#222', fg='white')
        size_label.grid(row=0, column=0, padx=5)

        size_options = [3, 4, 5]
        size_menu = ttk.Combobox(config_frame, textvariable=self.size_var, values=size_options, width=5)
        size_menu.grid(row=0, column=1, padx=5)
        size_menu.bind('<<ComboboxSelected>>', self.change_size)

        # Heuristic selection
        heuristic_label = tk.Label(config_frame, text="Heuristic:", bg='#222', fg='white')
        heuristic_label.grid(row=0, column=2, padx=5)

        manhattan_radio = tk.Radiobutton(
            config_frame, text="Manhattan", variable=self.heuristic_var,
            value="manhattan", bg='#222', fg='white', selectcolor='#444'
        )
        manhattan_radio.grid(row=0, column=3, padx=5)

        misplaced_radio = tk.Radiobutton(
            config_frame, text="Misplaced Tiles", variable=self.heuristic_var,
            value="misplaced", bg='#222', fg='white', selectcolor='#444'
        )
        misplaced_radio.grid(row=0, column=4, padx=5)

    def init_puzzle(self):
        self.puzzle = create_solvable_puzzle(self.size_var.get(), 100)

    def create_widgets(self):
        # Frame for the puzzle grid
        self.puzzle_frame = tk.Frame(self.root, bg='#222')
        self.puzzle_frame.pack(pady=10)

        # Create tile buttons
        self.buttons = []
        for i in range(self.size):
            row = []
            for j in range(self.size):
                val = self.puzzle.puzzle[i][j]
                btn = tk.Button(
                    self.puzzle_frame,
                    text=str(val) if val != 0 else '',
                    width=5, height=2,
                    font=('Arial', 18, 'bold'),
                    bg='#444',
                    fg='white',
                    activebackground='#666',
                    command=lambda r=i, c=j: self.tile_click(r, c)
                )
                btn.grid(row=i, column=j, padx=5, pady=5)
                row.append(btn)
            self.buttons.append(row)

        # Control frame
        control_frame = tk.Frame(self.root, bg='#222')
        control_frame.pack(pady=5)

        # Direction control buttons
        self.up_btn = tk.Button(control_frame, text='↑', command=lambda: self.move('U'), width=5, bg='#555', fg='white')
        self.up_btn.grid(row=0, column=1, pady=5)

        self.left_btn = tk.Button(control_frame, text='←', command=lambda: self.move('L'), width=5, bg='#555', fg='white')
        self.left_btn.grid(row=1, column=0, padx=5)

        self.down_btn = tk.Button(control_frame, text='↓', command=lambda: self.move('D'), width=5, bg='#555', fg='white')
        self.down_btn.grid(row=1, column=1)

        self.right_btn = tk.Button(control_frame, text='→', command=lambda: self.move('R'), width=5, bg='#555', fg='white')
        self.right_btn.grid(row=1, column=2, padx=5)

        # Action buttons frame
        action_frame = tk.Frame(self.root, bg='#222')
        action_frame.pack(pady=5)

        self.solve_btn = tk.Button(action_frame, text='🧠 Solve', command=self.solve_ai, bg='orange', width=15)
        self.solve_btn.grid(row=0, column=0, padx=5)

        self.reset_btn = tk.Button(action_frame, text='🔄 Reset', command=self.reset_puzzle, bg='#444', fg='white', width=15)
        self.reset_btn.grid(row=0, column=1, padx=5)

        # Stats frame
        stats_frame = tk.Frame(self.root, bg='#222')
        stats_frame.pack(pady=5)

        # Stats labels
        self.time_label = tk.Label(stats_frame, text="Time: 0.00s", bg='#222', fg='white', font=('Arial', 12), width=20)
        self.time_label.grid(row=0, column=0, pady=2)

        self.steps_label = tk.Label(stats_frame, text="Steps: 0", bg='#222', fg='white', font=('Arial', 12), width=20)
        self.steps_label.grid(row=1, column=0, pady=2)

        self.nodes_label = tk.Label(stats_frame, text="Nodes expanded: 0", bg='#222', fg='white', font=('Arial', 12), width=20)
        self.nodes_label.grid(row=2, column=0, pady=2)

        self.path_length_label = tk.Label(stats_frame, text="Solution length: 0", bg='#222', fg='white', font=('Arial', 12), width=20)
        self.path_length_label.grid(row=3, column=0, pady=2)

        # Comparison frame for heuristic results
        comparison_frame = tk.Frame(self.root, bg='#222')
        comparison_frame.pack(pady=5)

        self.comparison_label = tk.Label(
            comparison_frame,
            text="Heuristic Comparison",
            bg='#222', fg='white',
            font=('Arial', 12, 'bold'),
            width=40
        )
        self.comparison_label.pack(pady=2)

        self.comparison_results = tk.Text(
            comparison_frame,
            bg='#333', fg='white',
            height=6, width=40,
            font=('Courier', 10)
        )
        self.comparison_results.pack(pady=2)
        self.comparison_results.insert(tk.END, "Run the solver with different heuristics to compare.")

    def change_size(self, event=None):
        size = self.size_var.get()
        if size == self.size:
            return
        self.size = size
        # Clear existing puzzle
        for widget in self.puzzle_frame.winfo_children():
            widget.destroy()
        # Create new puzzle
        self.init_puzzle()
        # Recreate buttons
        self.create_puzzle_buttons()
        self.update_metrics()

    def create_puzzle_buttons(self):
        # Create new buttons for the puzzle grid
        self.buttons = []
        for i in range(self.size):
            row = []
            for j in range(self.size):
                val = self.puzzle.puzzle[i][j]
                btn = tk.Button(
                    self.puzzle_frame,
                    text=str(val) if val != 0 else '',
                    width=5, height=2,
                    font=('Arial', 18, 'bold'),
                    bg='#444',
                    fg='white',
                    activebackground='#666',
                    command=lambda r=i, c=j: self.tile_click(r, c)
                )
                btn.grid(row=i, column=j, padx=5, pady=5)
                row.append(btn)
            self.buttons.append(row)

    def tile_click(self, row, col):
        # Find which direction to move based on tile clicked
        zero_row, zero_col = self.puzzle.zero
        # Check if clicked tile is adjacent to the empty space
        if (row == zero_row and abs(col - zero_col) == 1) or (col == zero_col and abs(row - zero_row) == 1):
            # Determine direction to move tile
            if row < zero_row:
                self.move('U')  # Move tile up (empty space down)
            elif row > zero_row:
                self.move('D')  # Move tile down (empty space up)
            elif col < zero_col:
                self.move('L')  # Move tile left (empty space right)
            elif col > zero_col:
                self.move('R')  # Move tile right (empty space left)

    def move(self, direction):
        if self.puzzle.doMove(direction):
            self.update_grid()
            self.update_metrics()
            if self.puzzle.checkPuzzle():
                messagebox.showinfo("Solved!", "🎉 Puzzle Solved!")

    def update_grid(self):
        for i in range(self.size):
            for j in range(self.size):
                val = self.puzzle.puzzle[i][j]
                self.buttons[i][j].config(text=str(val) if val != 0 else '')

    def update_metrics(self):
        if self.steps_label:
            self.steps_label.config(text=f"Steps: {self.puzzle.steps}")
        if self.nodes_label:
            self.nodes_label.config(text=f"Nodes expanded: {self.puzzle.nodes_expanded}")
        if self.path_length_label:
            self.path_length_label.config(text=f"Solution length: {self.puzzle.path_length}")

    def handle_keypress(self, event):
        key = event.keysym.lower()
        if key == 'up' or key == 'w':
            self.move('U')
        elif key == 'left' or key == 'a':
            self.move('L')
        elif key == 'down' or key == 's':
            self.move('D')
        elif key == 'right' or key == 'd':
            self.move('R')

    def solve_ai(self):
        self.solve_btn.config(state=tk.DISABLED, text="Solving...")
        self.root.update()

        # Get current heuristic
        heuristic = self.heuristic_var.get()

        # Save initial puzzle state for comparison
        initial_puzzle = self.puzzle.clone()

        # First solution with current heuristic
        start_time = time.time()
        path = self.puzzle.solve(heuristic)
        solve_time = time.time() - start_time

        # Save the results
        current_results = {
            'heuristic': heuristic,
            'time': solve_time,
            'nodes': self.puzzle.nodes_expanded,
            'path_length': self.puzzle.path_length if path else 0
        }

        # Reset puzzle to initial state for comparison
        self.puzzle = initial_puzzle

        # Try the other heuristic
        other_heuristic = 'misplaced' if heuristic == 'manhattan' else 'manhattan'

        start_time = time.time()
        other_path = self.puzzle.solve(other_heuristic)
        other_solve_time = time.time() - start_time

        # Save other results
        other_results = {
            'heuristic': other_heuristic,
            'time': other_solve_time,
            'nodes': self.puzzle.nodes_expanded,
            'path_length': self.puzzle.path_length if other_path else 0
        }

        # Display comparison
        self.comparison_results.delete(1.0, tk.END)
        self.comparison_results.insert(tk.END, f"Comparison for {self.size}x{self.size} puzzle:\n")
        self.comparison_results.insert(tk.END, f"Manhattan: {current_results['time']:.3f}s, {current_results['nodes']} nodes\n")
        self.comparison_results.insert(tk.END, f"Misplaced: {other_results['time']:.3f}s, {other_results['nodes']} nodes\n")

        # Recommend better heuristic
        if current_results['time'] < other_results['time']:
            better = current_results['heuristic']
        else:
            better = other_results['heuristic']

        self.comparison_results.insert(tk.END, f"\n👉 {better.capitalize()} heuristic is better for this puzzle.")

        # Continue with selected heuristic
        self.puzzle = initial_puzzle
        path = current_results['heuristic'] == heuristic and path or other_path

        if path:
            self.time_label.config(text=f"Time to solve: {solve_time:.2f}s")
            self.animate_solution(path)
        else:
            messagebox.showerror("Unsolvable", "This puzzle cannot be solved.")
            self.solve_btn.config(state=tk.NORMAL, text="🧠 Solve")

    def animate_solution(self, path):
        if not path:
            if self.puzzle.checkPuzzle():
                messagebox.showinfo("AI", "✅ AI finished solving the puzzle!")
            self.solve_btn.config(state=tk.NORMAL, text="🧠 Solve")
            return
        move = path.pop(0)
        self.move(move)
        self.root.after(300, lambda: self.animate_solution(path))

    def reset_puzzle(self):
        new_puzzle = create_solvable_puzzle(self.size, 100)
        self.puzzle = new_puzzle
        self.puzzle.steps = 0
        self.puzzle.nodes_expanded = 0
        self.puzzle.path_length = 0
        self.update_grid()
        self.update_metrics()
        self.time_label.config(text="Time: 0.00s")
        self.comparison_results.delete(1.0, tk.END)
        self.comparison_results.insert(tk.END, "Run the solver with different heuristics to compare.")


def generate(size, moves):
    puzzle = [[(i * size + j + 1) % (size * size) for j in range(size)] for i in range(size)]
    p = TilePuzzle(puzzle)
    last = ''
    opp = {'L': 'R', 'R': 'L', 'U': 'D', 'D': 'U'}
    for _ in range(moves):
        possible = [m for m in p.moves if m != opp.get(last, '')]
        if not possible:
            possible = p.moves
        m = random.choice(possible)
        if p.doMove(m):
            last = m
    p.steps = 0
    return p


def is_solvable(puzzle):
    """Check if a puzzle is solvable using the inversion count method"""
    flat = []
    zero_row = 0
    for i, row in enumerate(puzzle):
        for val in row:
            if val == 0:
                zero_row = i
                continue
            flat.append(val)

    inversions = 0
    for i in range(len(flat)):
        for j in range(i+1, len(flat)):
            if flat[i] > flat[j]:
                inversions += 1

    # For odd-sized puzzles, solvable if inversions is even
    if len(puzzle) % 2 == 1:
        return inversions % 2 == 0
    # For even-sized puzzles, solvable if (inversions + row of zero) is odd
    else:
        return (inversions + zero_row) % 2 == 1


def create_solvable_puzzle(size, moves):
    """Generate a guaranteed solvable puzzle"""
    while True:
        p = generate(size, moves)
        if is_solvable(p.puzzle):
            return p


if __name__ == "__main__":
    root = tk.Tk()
    root.title("🧩 N-Puzzle Solver with AI")
    root.config(bg="#222")

    gui = PuzzleGUI(root)
    root.mainloop()