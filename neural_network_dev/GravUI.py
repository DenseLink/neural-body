import pygame
import math
import sys

pygame.init()


def main():
    # setting screen size
    scr_width = 1000
    scr_height = 700
    screen = pygame.display.set_mode((scr_width, scr_height))

    # setting caption
    pygame.display.set_caption("Elliptical orbit")

    # creating clock variable
    clock = pygame.time.Clock()

    # trackers for the planet trails
    num_planets = 5
    tail_length = 150
    orbits(screen, num_planets, tail_length, clock, scr_width, scr_height)


# simple function for putting the sun on the screen at a given coordinate. In the future, this style will be used for the
# planets and the png that keeps their dark side shadowed
def sun(screen, x, y):
    sunImg = pygame.image.load('sun.png')
    sunImg = pygame.transform.scale(sunImg, (70, 70))
    screen.blit(sunImg, (x, y))


def orbits(screen, num_planets, tail_length, clock, scr_width, scr_height):
    xtrack = [[0] * tail_length for i in range(num_planets)]
    ytrack = [[0] * tail_length for i in range(num_planets)]
    view = 0
    click_now = 0
    while (True):

        # setting x and y radius of circle (will always be equal unless we want an isometric view
        xRadius = 250
        yRadius = 250

        # just runs through a large number of sequential frames with bogus data
        for degree in range(0, 10000, 1):
            # the functions below will be replaced by neural net output
            x1 = int(math.cos(degree * 2 * math.pi / 360) * xRadius) + int(scr_width / 1.5)
            y1 = int(math.sin(degree * 2 * math.pi / 360) * yRadius) + int(scr_height / 2)

            x2 = int(math.cos(degree * 3 * math.pi / 360) * xRadius / 2.8) + int(scr_width / 1.5)
            y2 = int(math.sin(degree * 3 * math.pi / 360) * yRadius / 2.8) + int(scr_height / 2)

            x3 = int(math.cos(degree * 1.5 * math.pi / 360) * xRadius / 1.5) + int(scr_width / 1.5)
            y3 = int(math.sin(degree * 1.5 * math.pi / 360) * yRadius / 1.5) + int(scr_height / 2)

            x4 = int(math.cos(degree * 2.3 * math.pi / 360) * xRadius / 1.3) + int(scr_width / 1.5)
            y4 = int(math.sin(degree * 2.3 * math.pi / 360) * yRadius / 1.3) + int(scr_height / 2)

            x5 = int(math.cos(degree * 2.8 * math.pi / 360) * xRadius / 2) + int(scr_width / 1.5)
            y5 = int(math.sin(degree * 2.8 * math.pi / 360) * yRadius / 2) + int(scr_height / 2)

            # setting the stage
            screen.fill((0, 0, 0))
            #sun(screen, int(scr_width / 1.5) - 35, int(scr_height / 2) - 35)
            sunImg = pygame.image.load('sun.png')
            sunImg = pygame.transform.scale(sunImg, (70, 70))
            screen.blit(sunImg, (int(scr_width / 1.5) - 35, int(scr_height / 2) - 35))

            if view == 0:
                # iterates through the 2D list and draws the planet's trails
                for k in range(0, num_planets):
                    for j in range(1, tail_length - 1):
                        i = tail_length - j
                        if xtrack[k][j - 1] != 0:
                            pygame.draw.line(screen, (
                                255 - 255 * (i / tail_length), 255 - 255 * (i / tail_length),
                                255 - 255 * (i / tail_length)),
                                             [xtrack[k][j], ytrack[k][j]], [xtrack[k][j - 1], ytrack[k][j - 1]], 1)

                # draws the planets at their calculated locations
                pygame.draw.circle(screen, (255, 255, 0), [x2, y2], 5)
                pygame.draw.circle(screen, (0, 255, 255), [x3, y3], 5)
                pygame.draw.circle(screen, (255, 255, 255), [x4, y4], 5)
                pygame.draw.circle(screen, (255, 0, 0), [x5, y5], 5)
                pygame.draw.circle(screen, (0, 255, 0), [x1, y1], 5)
            else:
                pygame.draw.circle(screen, (255, 255, 0), [x2, int(scr_height / 2)], 5)
                pygame.draw.circle(screen, (0, 255, 255), [x3, int(scr_height / 2)], 5)
                pygame.draw.circle(screen, (255, 255, 255), [x4, int(scr_height / 2)], 5)
                pygame.draw.circle(screen, (255, 0, 0), [x5, int(scr_height / 2)], 5)
                pygame.draw.circle(screen, (0, 255, 0), [x1, int(scr_height / 2)], 5)

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
            view, click_now = menu(screen, view, clock, scr_width, scr_height, click_now)
            pygame.display.flip()

            clock.tick(60)  # screen refresh rate


def menu(screen, view, clock, scr_width, scr_height, click_now):
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        else:
            # compiler says this has no effect, but I think that is system-dependent.
            # the window was unmovable and unable to be exited until I added this.
            None
    pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(int(scr_width/30),int(scr_width/30),int(scr_width/3.5),int(scr_height/1.1)), 2)
    text_handler(screen, 'Menu', int(scr_width / 10), int(scr_height / 10), 40)

    if 150 + 100 > mouse[0] > 150 and 450 + 50 > mouse[1] > 450:
        pygame.draw.rect(screen, (0,255,0), (150, 450, 100, 50))
        # this click_now stuff makes it so that a held down mouse does not swap the view on every refresh
        if click[0] == 1:
            if click_now == 0:
                return abs(view - 1), 1
            else:
                return view, 1
        else:
            return view, 0
    else:
        return view, 0

def text_handler(screen, text, scr_x, scr_y, size):
    largeText = pygame.font.Font('freesansbold.ttf', size)
    textSurf = largeText.render(text, True, (255,255,255))
    textRect = textSurf.get_rect()
    textRect.center = (scr_x, scr_y)
    screen.blit(textSurf, textRect)

if __name__ == "__main__":
    main()
