import pygame

import math

import sys

pygame.init()

# setting screen size
scr_width = 600
scr_height = 600
screen = pygame.display.set_mode((scr_width, scr_height))

# setting caption
pygame.display.set_caption("Elliptical orbit")

# creating clock variable
clock = pygame.time.Clock()

# trackers for the planet trails
num_planets = 5
tail_length = 150
xtrack = [[0] * tail_length for i in range(num_planets)]
ytrack = [[0] * tail_length for i in range(num_planets)]
sunImg = pygame.image.load('sun.png')
sunImg = pygame.transform.scale(sunImg, (70, 70))


# simple function for putting the sun on the screen at a given coordinate. In the future, this style will be used for the
# planets and the png that keeps their dark side shadowed
def sun(x, y):
    screen.blit(sunImg, (x, y))

while (True):

    # setting x and y radius of circle (will always be equal unless we want an isometric view
    xRadius = 250
    yRadius = 250

    # just runs through a large number of sequential frames with bogus data
    for degree in range(0, 10000, 1):
        # the functions below will be replaced by neural net output
        x1 = int(math.cos(degree * 2 * math.pi / 360) * xRadius) + int(scr_width / 2)
        y1 = int(math.sin(degree * 2 * math.pi / 360) * yRadius) + int(scr_height / 2)

        x2 = int(math.cos(degree * 3 * math.pi / 360) * xRadius / 2.8) + int(scr_width / 2)
        y2 = int(math.sin(degree * 3 * math.pi / 360) * yRadius / 2.8) + int(scr_height / 2)

        x3 = int(math.cos(degree * 1.5 * math.pi / 360) * xRadius / 1.5) + int(scr_width / 2)
        y3 = int(math.sin(degree * 1.5 * math.pi / 360) * yRadius / 1.5) + int(scr_height / 2)

        x4 = int(math.cos(degree * 2.3 * math.pi / 360) * xRadius / 1.3) + int(scr_width / 2)
        y4 = int(math.sin(degree * 2.3 * math.pi / 360) * yRadius / 1.3) + int(scr_height / 2)

        x5 = int(math.cos(degree * 2.8 * math.pi / 360) * xRadius / 2) + int(scr_width / 2)
        y5 = int(math.sin(degree * 2.8 * math.pi / 360) * yRadius / 2) + int(scr_height / 2)

        # setting the stage
        screen.fill((0, 0, 0))
        sun(int(scr_width / 2) - 35, int(scr_height / 2) - 35)

        # iterates through the 2D list and draws the planet's trails
        for k in range(0, num_planets):
            for j in range(1, tail_length - 1):
                i = tail_length - j
                if xtrack[k][j - 1] != 0:
                    pygame.draw.line(screen, (
                        255 - 255 * (i / tail_length), 255 - 255 * (i / tail_length), 255 - 255 * (i / tail_length)),
                                     [xtrack[k][j], ytrack[k][j]], [xtrack[k][j - 1], ytrack[k][j - 1]], 1)

        # draws the planets at their calculated locations
        pygame.draw.circle(screen, (255, 255, 0), [x2, y2], 5)
        pygame.draw.circle(screen, (0, 255, 255), [x3, y3], 5)
        pygame.draw.circle(screen, (255, 255, 255), [x4, y4], 5)
        pygame.draw.circle(screen, (255, 0, 0), [x5, y5], 5)
        pygame.draw.circle(screen, (0, 255, 0), [x1, y1], 5)

        # shifts all data points within the lists to the left to make room for the new trail data point
        for j in range(0, num_planets):
            for i in range(0, tail_length - 1):
                xtrack[j][i] = xtrack[j][i + 1]
                ytrack[j][i] = ytrack[j][i + 1]

        # pushing the new trail x and y datapoints to the back of the lists
        # the x1...y5 values will be replaced with two 2D lists when real data is used, making this 4 lines
        xtrack[0][tail_length - 1] = x1
        xtrack[1][tail_length - 1] = x2
        xtrack[2][tail_length - 1] = x3
        xtrack[3][tail_length - 1] = x4
        xtrack[4][tail_length - 1] = x5
        ytrack[0][tail_length - 1] = y1
        ytrack[1][tail_length - 1] = y2
        ytrack[2][tail_length - 1] = y3
        ytrack[3][tail_length - 1] = y4
        ytrack[4][tail_length - 1] = y5

        # updates the display with the new frame
        pygame.display.flip()

        # handles events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            else:
                # compiler says this has no effect, but I think that is system-dependent.
                # the window was unmovable and unable to be exited until I added this.
                None

        clock.tick(60)  # screen refresh rate
