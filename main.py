import pygame
from queue import PriorityQueue
import numpy as np
from numpy.random import random_integers as rnd
import math
import random
import copy

class Game:
    def __init__(self):
        pygame.init()
        pygame.font.init()
        pygame.display.set_caption("Simple Path Finding Agent")
        self.clock = pygame.time.Clock()
        font = pygame.font.Font("assets/Montserrat-SemiBold.ttf", 30)
        #font.set_bold(True)
        font_thetastar = pygame.font.Font("assets/Montserrat-SemiBold.ttf", 22)
        #font_thetastar.set_bold(True)
        self.text = font.render("Algorithms", True, Color.WHITE)
        self.text_astar = font.render("A*", True, Color.WHITE)
        self.text_thetastar = font_thetastar.render("Theta*", True, Color.WHITE)
        self.text_GA = font.render("GA", True, Color.WHITE)
        self.text_comp = font.render("Complexity", True, Color.WHITE)
        self.text_25 = font.render("25", True, Color.WHITE)
        self.text_50 = font.render("50", True, Color.WHITE)
        self.text_100 = font.render("100", True, Color.WHITE)
        self.text_barrier = font.render("100", True, Color.WHITE)
        self.text_bmaze = font.render("Auto", True, Color.WHITE)
        self.text_bmouse = font.render("Mouse", True, Color.WHITE)
        self.text_MG = font.render("Maze Generation", True, Color.WHITE)
        self.text_reset = font.render("Reset", True, Color.WHITE)
        self.text_play = font.render("Play", True, Color.WHITE)
        self.WIDTH = 920
        self.HEIGHT = 600
        self.WIN = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.WIN.fill(Color.DARK_GRAY)
        self.start = None
        self.end = None
        self.bstarted = False
        self.bgenerated = False
        self.bmouse = True
        self.brun = True
        self.bastar = True
        self.btheta_star = False
        self.bGA = False
        self.b25 = True
        self.b50 = False
        self.b100 = False
        self.rows = 25
        self.grid = self.construct_gridboard(self.rows, self.HEIGHT)
        self.button_astar = pygame.Rect(610, 70, 90, 50)
        self.button_thetastar = pygame.Rect(715, 70, 90, 50)
        self.button_GA = pygame.Rect(820, 70, 90, 50)
        self.button_s25 = pygame.Rect(610, 200, 90, 50)
        self.button_s50 = pygame.Rect(715, 200, 90, 50)
        self.button_s100 = pygame.Rect(820, 200, 90, 50)
        self.button_bmaze = pygame.Rect(765, 320, 145, 50)
        self.button_bmouse = pygame.Rect(610, 320, 145, 50)
        self.button_reset = pygame.Rect(620, 400, 280, 50)
        self.button_play = pygame.Rect(620, 460, 280, 50)
        pygame.draw.rect(self.WIN, Color.LIGHT_GRAY, self.button_astar, 0, 10)
        pygame.draw.rect(self.WIN, Color.GRAY, self.button_thetastar, 0, 10)
        pygame.draw.rect(self.WIN, Color.GRAY, self.button_GA, 0, 10)
        pygame.draw.rect(self.WIN, Color.LIGHT_GRAY, self.button_s25, 0, 10)
        pygame.draw.rect(self.WIN, Color.GRAY, self.button_s50, 0, 10)
        pygame.draw.rect(self.WIN, Color.GRAY, self.button_s100, 0, 10)
        pygame.draw.rect(self.WIN, Color.GRAY, self.button_bmaze, 0, 10)
        pygame.draw.rect(self.WIN, Color.LIGHT_GRAY, self.button_bmouse, 0, 10)
        pygame.draw.rect(self.WIN, Color.GRAY, (610, 390, 300, 130), 0, 10)
        pygame.draw.rect(self.WIN, Color.RED, self.button_reset, 0, 10)
        pygame.draw.rect(self.WIN, Color.LIGHT_GRAY, self.button_play, 0, 10)
        self.WIN.blit(self.text, (610, 15))
        self.WIN.blit(self.text_astar, (635, 75))
        self.WIN.blit(self.text_thetastar, (723, 79))
        self.WIN.blit(self.text_GA, (840, 75))
        self.WIN.blit(self.text_comp, (610, 145))
        self.WIN.blit(self.text_25, (633, 205))
        self.WIN.blit(self.text_50, (735, 205))
        self.WIN.blit(self.text_100, (837, 205))
        self.WIN.blit(self.text_MG, (610, 265))
        self.WIN.blit(self.text_bmouse, (630, 324))
        self.WIN.blit(self.text_bmaze, (800, 324))
        self.WIN.blit(self.text_reset, (717, 404))
        self.WIN.blit(self.text_play, (725, 464))

    def maze(self, width, height, complexity=.75, density=.9):
        # Only odd shapes
        shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
        # Adjust complexity and density relative to maze size
        complexity = int(complexity * (5 * (shape[0] + shape[1])))
        density = int(density * (shape[0] // 2 * shape[1] // 2))
        # Build actual maze
        grid = np.zeros(shape, dtype=bool)
        # Fill borders
        grid[0, :] = grid[-1, :] = 1
        grid[:, 0] = grid[:, -1] = 1
        # Make isles
        for i in range(density):
            x, y = (rnd(0, shape[1] / 2) * 2), rnd(0, shape[0] / 2) * 2
            grid[y, x] = 1
            for j in range(complexity):
                neighbours = []
                if x > 1:           neighbours.append((y, x - 2))
                if x < shape[1] - 2:  neighbours.append((y, x + 2))
                if y > 1:           neighbours.append((y - 2, x))
                if y < shape[0] - 2:  neighbours.append((y + 2, x))
                if len(neighbours):
                    y_, x_ = neighbours[rnd(0, len(neighbours) - 1)]
                    if grid[y_, x_] == 0:
                        grid[y_, x_] = 1
                        grid[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                        x, y = x_, y_
        return grid

    def maze2(self, width, height):
        pass


    def construct_gridboard(self, rows, height):
        grid = []
        gap = height // rows
        for row in range(rows):
            grid.append([])
            for column in range(rows):
                square = Square(row, column, gap, rows)
                grid[row].append(square)

        return grid

    def draw_gridlines(self, win, rows, height):
        gap = height // rows
        for i in range(rows):
            pygame.draw.line(win, Color.BLACK, (0, i * gap), (height, i * gap))
            for j in range(rows + 1):
                pygame.draw.line(win, Color.BLACK, (j * gap, 0), (j * gap, height))

    def draw(self, win, grid, rows, height):
        for row in grid:
            for square in row:
                square.draw(win)
        self.draw_gridlines(win, rows, height)
        pygame.display.update()

    def get_click_pos(self, pos, rows, height):
        gap = height // rows
        y, x = pos

        row = y // gap
        col = x // gap

        return row, col

    def reset_gridboard(self):
        self.start = None
        self.end = None
        self.grid = self.construct_gridboard(self.rows, self.HEIGHT)
        self.bstarted = False
        self.bgenerated = False

    def running(self):
        while self.brun:
            self.draw(self.WIN, self.grid, self.rows, self.HEIGHT)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.brun = False

                if pygame.mouse.get_pressed()[0]:  # LEFT
                    pos = pygame.mouse.get_pos()
                    x, y = pos
                    if self.button_astar.x < x < self.button_astar.x + self.button_astar.width and self.button_astar.y < y < self.button_astar.y + self.button_astar.height:
                        self.bastar = True
                        self.btheta_star = False
                        self.bGA = False
                        pygame.draw.rect(self.WIN, Color.LIGHT_GRAY, self.button_astar, 0, 10)
                        pygame.draw.rect(self.WIN, Color.GRAY, self.button_thetastar, 0, 10)
                        pygame.draw.rect(self.WIN, Color.GRAY, self.button_GA, 0, 10)
                        self.WIN.blit(self.text_astar, (635, 75))
                        self.WIN.blit(self.text_thetastar, (723, 79))
                        self.WIN.blit(self.text_GA, (840, 75))
                    if self.button_thetastar.x < x < self.button_thetastar.x + self.button_thetastar.width and self.button_thetastar.y < y < self.button_thetastar.y + self.button_thetastar.height:
                        self.bastar = False
                        self.btheta_star = True
                        self.bGA = False
                        pygame.draw.rect(self.WIN, Color.GRAY, self.button_astar, 0, 10)
                        pygame.draw.rect(self.WIN, Color.LIGHT_GRAY, self.button_thetastar, 0, 10)
                        pygame.draw.rect(self.WIN, Color.GRAY, self.button_GA, 0, 10)
                        self.WIN.blit(self.text_astar, (635, 75))
                        self.WIN.blit(self.text_thetastar, (723, 79))
                        self.WIN.blit(self.text_GA, (840, 75))
                    if self.button_GA.x < x < self.button_GA.x + self.button_GA.width and self.button_GA.y < y < self.button_GA.y + self.button_GA.height:
                        self.bastar = False
                        self.btheta_star = False
                        self.bGA = True
                        pygame.draw.rect(self.WIN, Color.GRAY, self.button_astar, 0, 10)
                        pygame.draw.rect(self.WIN, Color.GRAY, self.button_thetastar, 0, 10)
                        pygame.draw.rect(self.WIN, Color.LIGHT_GRAY, self.button_GA, 0, 10)
                        self.WIN.blit(self.text_astar, (635, 75))
                        self.WIN.blit(self.text_thetastar, (723, 79))
                        self.WIN.blit(self.text_GA, (840, 75))

                    if self.button_s25.x < x < self.button_s25.x + self.button_s25.width and self.button_s25.y < y < self.button_s25.y + self.button_s25.height and not self.bstarted:
                        self.b25 = True
                        self.b50 = False
                        self.b100 = False
                        pygame.draw.rect(self.WIN, Color.LIGHT_GRAY, self.button_s25, 0, 10)
                        pygame.draw.rect(self.WIN, Color.GRAY, self.button_s50, 0, 10)
                        pygame.draw.rect(self.WIN, Color.GRAY, self.button_s100, 0, 10)
                        self.WIN.blit(self.text_25, (633, 205))
                        self.WIN.blit(self.text_50, (735, 205))
                        self.WIN.blit(self.text_100, (837, 205))
                        self.rows = 25
                        self.reset_gridboard()

                    if self.button_s50.x < x < self.button_s50.x + self.button_s50.width and self.button_s50.y < y < self.button_s50.y + self.button_s50.height and not self.bstarted:
                        self.b25 = False
                        self.b50 = True
                        self.b100 = False
                        pygame.draw.rect(self.WIN, Color.GRAY, self.button_s25, 0, 10)
                        pygame.draw.rect(self.WIN, Color.LIGHT_GRAY, self.button_s50, 0, 10)
                        pygame.draw.rect(self.WIN, Color.GRAY, self.button_s100, 0, 10)
                        self.WIN.blit(self.text_25, (633, 205))
                        self.WIN.blit(self.text_50, (735, 205))
                        self.WIN.blit(self.text_100, (837, 205))
                        self.rows = 50
                        self.reset_gridboard()

                    if self.button_s100.x < x < self.button_s100.x + self.button_s100.width and self.button_s100.y < y < self.button_s100.y + self.button_s100.height and not self.bstarted:
                        self.b25 = False
                        self.b50 = False
                        self.b100 = True
                        pygame.draw.rect(self.WIN, Color.GRAY, self.button_s25, 0, 10)
                        pygame.draw.rect(self.WIN, Color.GRAY, self.button_s50, 0, 10)
                        pygame.draw.rect(self.WIN, Color.LIGHT_GRAY, self.button_s100, 0, 10)
                        self.WIN.blit(self.text_25, (633, 205))
                        self.WIN.blit(self.text_50, (735, 205))
                        self.WIN.blit(self.text_100, (837, 205))
                        self.rows = 100
                        self.reset_gridboard()

                    if self.button_bmouse.x < x < self.button_bmouse.x + self.button_bmouse.width and self.button_bmouse.y < y < self.button_bmouse.y + self.button_bmouse.height:
                        self.bmouse = True
                        pygame.draw.rect(self.WIN, Color.GRAY, self.button_bmaze, 0, 10)
                        pygame.draw.rect(self.WIN, Color.LIGHT_GRAY, self.button_bmouse, 0, 10)
                        self.WIN.blit(self.text_bmouse, (630, 324))
                        self.WIN.blit(self.text_bmaze, (800, 324))

                    if self.button_bmaze.x < x < self.button_bmaze.x + self.button_bmaze.width and self.button_bmaze.y < y < self.button_bmaze.y + self.button_bmaze.height:
                        self.bmouse = False
                        pygame.draw.rect(self.WIN, Color.LIGHT_GRAY, self.button_bmaze, 0, 10)
                        pygame.draw.rect(self.WIN, Color.GRAY, self.button_bmouse, 0, 10)
                        self.WIN.blit(self.text_bmouse, (630, 324))
                        self.WIN.blit(self.text_bmaze, (800, 324))

                    if self.button_play.x < x < self.button_play.x + self.button_play.width and self.button_play.y < y < self.button_play.y + self.button_play.height and self.start and self.end and not self.bstarted:
                        if self.bastar:
                            for row in self.grid:
                                for square in row:
                                    square.update_neighbors(self.grid)

                            Algorithm.astar(lambda: self.draw(self.WIN, self.grid, self.rows, self.HEIGHT), self.grid,
                                            self.start, self.end)

                            self.bstarted = True

                        elif self.btheta_star:
                            for row in self.grid:
                                for square in row:
                                    square.update_neighbors(self.grid)

                            Algorithm.theta_star(lambda: self.draw(self.WIN, self.grid, self.rows, self.HEIGHT),
                                                 self.grid,
                                                 self.start, self.end, self.WIN)
                            self.bstarted = True

                        elif self.bGA:
                            '''
                            Algorithm.GA(lambda: self.draw(self.WIN, self.grid, self.rows, self.HEIGHT),
                                                 self.grid,
                                                 self.start, self.end, self.WIN)
                            self.bstarted = True
                            '''
                            pass

                    if self.button_reset.x < x < self.button_reset.x + self.button_reset.width and self.button_reset.y < y < self.button_reset.y + self.button_reset.height:
                        self.reset_gridboard()
                        if not self.bmouse and not self.bgenerated:
                            maze = self.maze(self.rows, self.rows)
                            for i in range(self.rows):
                                for j in range(self.rows):
                                    if maze[i][j]:
                                        self.grid[i][j].make_barrier()
                            self.bgenerated = True

                    if x < self.HEIGHT:
                        row, col = self.get_click_pos(pos, self.rows, self.HEIGHT)
                        square = self.grid[row][col]
                        if not self.start and square != self.end and not square.is_barrier():
                            self.start = square
                            self.start.make_start()

                        elif not self.end and square != self.start and not square.is_barrier():
                            self.end = square
                            self.end.make_end()

                        elif square != self.end and square != self.start and not self.bgenerated and self.bmouse and not self.bstarted:
                            square.make_barrier()

                elif pygame.mouse.get_pressed()[2] and not self.bstarted:  # RIGHT
                    pos = pygame.mouse.get_pos()
                    x, y = pos
                    if x < self.HEIGHT:
                        row, col = self.get_click_pos(pos, self.rows, self.HEIGHT)
                        square = self.grid[row][col]
                        square.reset()
                        if square == self.start:
                            self.start = None
                        elif square == self.end:
                            self.end = None
                else:
                    pass
            pygame.display.update()
            pygame.display.flip()
            self.clock.tick(100)


class Color:
    SILVER = (192, 192, 192)
    LIGHT_GRAY = (111, 119, 131)
    WHITE = (255, 255, 255)
    BLACK = (37, 38, 43)
    PURPLE = (135, 132, 229)
    GRAY = (54, 57, 66)
    TURQUOISE = (200, 224, 208)
    RED = (186, 78, 83)
    DARK_GRAY = (37, 38, 43)
    MID_LIGHT_GRAY = (138, 145, 157)


class Square:
    def __init__(self, row, col, height, total_rows):
        self.row = row  # Current square row
        self.col = col  # Current square column
        self.x = row * height  # square x position
        self.y = col * height  # square y position
        self.color = Color.GRAY  # initial square color.
        self.neighbors = []  # the square node neighbors list.
        self.width = height  # square width
        self.total_rows = total_rows  # number of rows in the grid.

    def get_pos(self):
        return self.row, self.col

    def is_expanded(self):
        return self.color == Color.MID_LIGHT_GRAY

    def is_open(self):
        return self.color == Color.LIGHT_GRAY

    def is_barrier(self):
        return self.color == Color.BLACK

    def is_start(self):
        return self.color == Color.RED

    def is_end(self):
        return self.color == Color.TURQUOISE

    def reset(self):
        self.color = Color.GRAY

    def make_start(self):
        self.color = Color.RED

    def make_expanded(self):
        self.color = Color.MID_LIGHT_GRAY

    def make_open(self):
        self.color = Color.LIGHT_GRAY

    def make_barrier(self):
        self.color = Color.BLACK

    def make_end(self):
        self.color = Color.TURQUOISE

    def make_path(self):
        self.color = Color.PURPLE

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

    def update_neighbors(self, grid):
        self.neighbors = []  # Current node neighbors.
        if self.row < self.total_rows - 1 and not grid[self.row + 1][
            self.col].is_barrier():  # Checking if the bottom neighbor is not a barrier ...
            self.neighbors.append(grid[self.row + 1][self.col])  # Add to neighbors list.

        if self.row > 0 and not grid[self.row - 1][
            self.col].is_barrier():  # Checking if the top neighbor is not a barrier ...
            self.neighbors.append(grid[self.row - 1][self.col])  # Add to neighbors list.

        if self.col < self.total_rows - 1 and not grid[self.row][
            self.col + 1].is_barrier():  # Checking if the right neighbor is not a barrier ...
            self.neighbors.append(grid[self.row][self.col + 1])  # Add to neighbors list.

        if self.col > 0 and not grid[self.row][
            self.col - 1].is_barrier():  # Checking if the left neighbor is not a barrier ...
            self.neighbors.append(grid[self.row][self.col - 1])  # Add to neighbors list.

    def __lt__(self, other):
        return False


class Algorithm:

    def h(p1, p2):
        # Finding the heuristic value of A* and theta* algorithm by Euclidean formula.
        x1, y1 = p1
        x2, y2 = p2
        return math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

    def astar(draw, grid, start, end):
        count = 0  # Tie breaker
        open_set = PriorityQueue()
        open_set.put((0, count, start))  # (f_score, insertion number, node)
        parent = {}  # Keep track of the parents.
        g_score = {square: float("inf") for row in grid for square in
                   row}  # Initialize every node's g score to infinity.
        g_score[start] = 0  # initialize the g to be Zero for the start node.
        f_score = {square: float("inf") for row in grid for square in
                   row}  # Initialize every node's f score to infinity.
        f_score[start] = Algorithm.h(start.get_pos(), end.get_pos())  # initialize the f to be = h for the start node.
        open_set_hash = {start}  # To keep tracking of the node in the open set.
        while not open_set.empty():  # if the open set is not empty
            for event in pygame.event.get():
                if event.type == pygame.QUIT:  # if the user want to quit the program while running the algorithm.
                    pygame.quit()
            current = open_set.get()[2]  # removing the current node from the queue.
            open_set_hash.remove(current)
            if current == end:  # found the path
                Algorithm.reconstruct_path(parent, end, draw)  # draw the path from the start to the end.
                end.make_start()
                return True
            for neighbor in current.neighbors:
                temp_g_score = g_score[current] + 1  # g score of the neighbor is g for parent + 1
                if temp_g_score < g_score[neighbor]:  # if the new g score of the neighbor is less than before, then:
                    parent[neighbor] = current  # Switch the previous parent of the neighbor to the current node.
                    g_score[neighbor] = temp_g_score  # Update the g score to the new one
                    f_score[neighbor] = temp_g_score + Algorithm.h(neighbor.get_pos(),
                                                                   end.get_pos())  # neighbor f = g + h
                    if neighbor not in open_set_hash:  # If the neighbor is not in the open set
                        count += 1
                        open_set.put((f_score[neighbor], count, neighbor))  # Add the neighbor to the open set.
                        open_set_hash.add(neighbor)
                        neighbor.make_open()  # Change the status into generated neighbor. (green color)
            draw()
            if current != start:
                current.make_expanded()
        return False

    def reconstruct_path(parent, current, draw):
        while current in parent and current != parent[current]:
            current = parent[current]
            current.make_path()
            draw()

    def theta_star(draw, grid, start, end, win):
        open_set = PriorityQueue()
        parent = {}
        parent[start] = start
        g_score = {}
        g_score[start] = 0
        f_score = {square: float("inf") for row in grid for square in row}
        f_score[start] = Algorithm.h(start.get_pos(), end.get_pos())
        open_set.put((Algorithm.h(start.get_pos(), end.get_pos()), start))
        open_set_hash = {start}
        closed_set = {None}
        while not open_set.empty():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
            s = open_set.get()[1]
            open_set_hash.remove(s)
            closed_set.add(s)
            if s == end:
                Algorithm.reconstruct_path(parent, s, draw)
                end.make_start()
                return True
            for s1 in s.neighbors:
                if s1 not in closed_set:
                    if s1 not in open_set_hash:
                        f_score[s1] = Algorithm.h(s1.get_pos(), end.get_pos())
                        g_score[s1] = float("inf")
                        parent[s1] = None
                    Algorithm.update_vertex(grid, s, s1, g_score, f_score, open_set, open_set_hash, parent, end)
                    parent[s1] = s

            draw()
            if s != start:
                s.make_expanded()
        return False

    def line_of_sight(grid, s, s1):
        (x0, y0) = s.get_pos()
        (x1, y1) = s1.get_pos()
        dx = x1 - x0
        dy = y1 - y0
        f = 0
        if dy < 0:
            dy = -dy
            sy = -1
        else:
            sy = 1

        if dx < 0:
            dx = -dx
            sx = -1
        else:
            sx = 1

        if dx >= dy:
            while x0 != x1:
                f = f + dy
                if f >= dx:
                    if grid[x0 + int((sx - 1) / 2)][y0 + int((sy - 1) / 2)].is_barrier():
                        return False
                    y0 = y0 + sy
                    f = f - dx
                if f != 0 and grid[x0 + int((sx - 1) / 2)][y0 + int((sy - 1) / 2)].is_barrier():
                    return False
                if dy == 0 and grid[x0 + int((sx - 1) / 2)][y0].is_barrier() and grid[
                    x0 + int((sx - 1) / 2)][y0 - 1].is_barrier():
                    return False
                x0 = x0 + sx
        else:
            while y0 != y1:
                f = f + dx
                if f >= dy:
                    if grid[x0 + int((sx - 1) / 2)][y0 + int((sy - 1) / 2)].is_barrier():
                        return False
                    x0 = x0 + sx
                    f = f - dy
                if f != 0 and grid[x0 + int((sx - 1) / 2)][y0 + int((sy - 1) / 2)].is_barrier():
                    return False
                if dx == 0 and grid[x0][y0 + int((sy - 1) / 2)].is_barrier() and grid[
                    x0 - 1, y0 + int((sy - 1) / 2)].is_barrier():
                    return False
                y0 = y0 + sy
        return True

    def update_vertex(grid, s, s1, g_score, f_score, open_set, open_set_hash, parent, end):
        if Algorithm.line_of_sight(grid, parent[s], s1):
            if g_score[parent[s]] + Algorithm.h(parent[s].get_pos(), s1.get_pos()) < g_score[s1]:
                g_score[s1] = g_score[parent[s]] + Algorithm.h(parent[s].get_pos(), s1.get_pos())
                parent[s1] = parent[s]
                if s1 in open_set_hash:
                    s1 = open_set.get()[1]
                    open_set_hash.remove(s1)
                f_score[s1] = g_score[s1] + Algorithm.h(s1.get_pos(), end.get_pos())
                open_set.put((f_score[s1], s1))
                open_set_hash.add(s1)
                s1.make_open()
        else:
            if g_score[s] + Algorithm.h(s.get_pos(), s1.get_pos()) < g_score[s1]:
                g_score[s1] = g_score[s] + Algorithm.h(parent[s].get_pos(), s1.get_pos())
                parent[s1] = s
                if s1 in open_set_hash:
                    s1 = open_set.get()[1]
                    open_set_hash.remove(s1)
                f_score[s1] = g_score[s1] + Algorithm.h(s1.get_pos(), end.get_pos())
                open_set.put((f_score[s1], s1))
                open_set_hash.add(s1)
                s1.make_open()

    #To be implemented
    def GA(draw, grid, start, end, win):
        pass


def main():
    game = Game()
    game.running()
    pygame.quit()


if __name__ == "__main__":
    main()
