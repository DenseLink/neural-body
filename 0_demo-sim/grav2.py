import pygame
import math
import sys
import pandas as pd
from BenrulesRealTimeSim import BenrulesRealTimeSim
import os

# Set audio driver to avoid ALSA errors
os.environ['SDL_AUDIODRIVER'] = 'dsp'

# Check if DISPLAY has been detected.  If not, assume WSL with pycharm and grab
# display connection.
# Only needed for launching from pycharm
try:
    print(os.environ["DISPLAY"])
except KeyError as error:
    # If at this point, DISPLAY doesn't exist and needs to be set.
    line_list = []
    with open('/etc/resolv.conf') as f:
        for line in f:
            pass
        last_line = line
        line_list = last_line.split(' ')
        f.close()
    # Set the display
    os.environ["DISPLAY"] = line_list[-1].rstrip() + ":0"

pygame.init()


def main():
    # Set screen size and launch window
    scr_width = 1000
    scr_height = 700
    screen = pygame.display.set_mode((scr_width, scr_height))

    # Setting caption at top of window
    pygame.display.set_caption("AstroGators - Neural Body Simulator")

    # Creating clock variable.
    # Helps with tracking time and managing framerate.
    clock = pygame.time.Clock()

    # trackers for the planet trails
    num_planets = 13  #TODO: Initialize the number of bodies from the config file.  Are we just adding inner view as other planets?
    tail_length = 150
    orbits(screen, num_planets, tail_length, clock, scr_width, scr_height)


# simple function for putting the sun on the screen at a given coordinate. In the future, this style will be used for
# the planets and the png that keeps their dark side shadowed
def sun(screen, x, y):
    """
    A simple function for putting the sun on the screen at a given coordinate.

    TODO: In the future, this style will be used for the planets and the png that keeps their dark side shadowed.

    Args:
        screen: Output display.
        x:
        y:

    Returns:

    """
    sun_img = pygame.image.load('sun.png')
    sun_img = pygame.transform.scale(sun_img, (10, 10))
    screen.blit(sun_img, (x, y))


def orbits(screen, num_planets, tail_length, clock, scr_width, scr_height):
    x_track = [[0] * tail_length for i in range(num_planets)]
    y_track = [[0] * tail_length for i in range(num_planets)]

    pause = 0
    view = 0
    click_now = 0
    input_active = 0
    textbox_active = 0
    input_text = ""

    # Factor to setup view magnification for all planets.
    # Ranges from 15 to show all planets down to 1 to show only through Mars.
    # Zooms are later used for scaling views from simulation space to the view
    # of the inner planets and view of the outer planet.
    zoom_factor = 15
    zoom = 1000000000 * zoom_factor
    # Set inner planet view zoom
    zoom_i = 15000000000
    # Set time step and speed factor.
    # 1 is about a 35 second year (earth)
    speed = 1
    # TODO: Figure out time step
    # time_step = 750000 * speed
    time_step = 118800 * speed
    # Keep track of how many simulation time steps have passed.
    curr_time_step = 0

    # Create simulation object that keeps track of simultion state and predicts
    # position of satellite specified in config file.
    # TODO: Get CSV location from prompt and load CSV file to pandas dataframe.
    # TODO: Add error handling for file opening.
    # TODO: Add error handling that check to make sure pluto or mars are the only "satellites" selected.  Make sure it is a valid config file.
    simulation = BenrulesRealTimeSim(
        time_step=time_step,
        in_config_df = pd.read_csv("mars_sim_config.csv")
    )
    # Set offset so objects orbit the correct point.
    sunx = int(scr_width / 1.5)
    suny = int(scr_height / 2)

    sun_i_x = 185
    sun_i_y = 520
    while True:
        # just runs through a large number of sequential frames with bogus data
        # TODO: What does this mean?
        degree = 0

        # Get next simulator state (positioning of all objects).
        current_positions, predicted_position = simulation.get_next_sim_state()

        while degree < 10000:
            # the functions below will be replaced by neural net output
            curr_time_step += 1

            if speed == 0.5 and pause == 0:
                # Only advance simulation every other degree.
                if degree % 2 == 0:
                    current_positions, predicted_position = simulation.get_next_sim_state()

            if speed == 1 and pause == 0:
                # Advance simulation every degree
                current_positions, predicted_position = simulation.get_next_sim_state()

            if speed == 2 and pause == 0:
                # Advance simulation twice per degree
                current_positions, predicted_position = simulation.get_next_sim_state()
                current_positions, predicted_position = simulation.get_next_sim_state()

            if speed == 4 and pause == 0:
                # Advance simulation 4 times every degree
                current_positions, predicted_position = simulation.get_next_sim_state()
                current_positions, predicted_position = simulation.get_next_sim_state()
                current_positions, predicted_position = simulation.get_next_sim_state()
                current_positions, predicted_position = simulation.get_next_sim_state()

            # Calculate the relative position of each body to the sun.
            x1 = int((current_positions['mercury'][0] - current_positions['sun'][0]) / zoom) + sunx
            y1 = int((current_positions['mercury'][1] - current_positions['sun'][1]) / zoom) + suny
            xi1 = int((current_positions['mercury'][0] - current_positions['sun'][0])/ zoom_i*8) + sun_i_x
            yi1 = int((current_positions['mercury'][1] - current_positions['sun'][1])/ zoom_i*8) + sun_i_y
            # print("x1:", x1, "...y1:", y1)

            x2 = int((current_positions['venus'][0] - current_positions['sun'][0]) / zoom) + sunx
            y2 = int((current_positions['venus'][1] - current_positions['sun'][1]) / zoom) + suny
            xi2 = int((current_positions['venus'][0] - current_positions['sun'][0]) / zoom_i*8) + sun_i_x
            yi2 = int((current_positions['venus'][1] - current_positions['sun'][1]) / zoom_i*8) + sun_i_y
            # print("x2:", x2, "...y2:", y2)

            x3 = int((current_positions['earth'][0] - current_positions['sun'][0]) / zoom) + sunx
            y3 = int((current_positions['earth'][1] - current_positions['sun'][1]) / zoom) + suny
            xi3 = int((current_positions['earth'][0] - current_positions['sun'][0]) / zoom_i*8) + sun_i_x
            yi3 = int((current_positions['earth'][1] - current_positions['sun'][1]) / zoom_i*8) + sun_i_y
            # print("x3:", x3, "...y3:", y3)

            x4 = int((current_positions['mars'][0] - current_positions['sun'][0]) / zoom) + sunx
            y4 = int((current_positions['mars'][1] - current_positions['sun'][1]) / zoom) + suny
            xi4 = int((current_positions['mars'][0] - current_positions['sun'][0]) / zoom_i*8) + sun_i_x
            yi4 = int((current_positions['mars'][1] - current_positions['sun'][1]) / zoom_i*8) + sun_i_y

            x5 = int((current_positions['jupiter'][0] - current_positions['sun'][0]) / zoom) + sunx
            y5 = int((current_positions['jupiter'][1] - current_positions['sun'][1]) / zoom) + suny

            x6 = int((current_positions['saturn'][0] - current_positions['sun'][0]) / zoom) + sunx
            y6 = int((current_positions['saturn'][1] - current_positions['sun'][1]) / zoom) + suny

            x7 = int((current_positions['uranus'][0] - current_positions['sun'][0]) / zoom) + sunx
            y7 = int((current_positions['uranus'][1] - current_positions['sun'][1]) / zoom) + suny

            x8 = int((current_positions['neptune'][0] - current_positions['sun'][0]) / zoom) + sunx
            y8 = int((current_positions['neptune'][1] - current_positions['sun'][1]) / zoom) + suny

            if 'pluto' in predicted_position:
                x9 = int((predicted_position['pluto'][0] - current_positions['sun'][0]) / zoom) + sunx
                y9 = int((predicted_position['pluto'][1] - current_positions['sun'][1]) / zoom) + suny
            else:
                x9 = 0
                y9 = 0

            # Setting the stage.
            screen.fill((0, 0, 0))
            # Load and position the sun.
            # TODO: Why are we loading the sun image every single frame?  Why not cache in memory outside the loop?
            sun_img = pygame.image.load('sun.png')
            sun_img = pygame.transform.scale(sun_img, (10, 10))
            screen.blit(sun_img, (int(scr_width / 1.5) - 5, int(scr_height / 2) - 5))
            screen.blit(sun_img, (sun_i_x-5, sun_i_y-5))

            # If simulation not paused, then continue with moving the objects.
            if pause == 0:
                # TODO: Could a queue just be used here?
                # shifts all data points within the lists to the left to make room for the new trail data point
                for j in range(0, num_planets):
                    for i in range(0, tail_length - 1):
                        x_track[j][i] = x_track[j][i + 1]
                        y_track[j][i] = y_track[j][i + 1]

                # pushing the new trail x and y datapoints to the back of the lists
                # the x1...y5 values will be replaced with two 2D lists when real data is used, making this 4 lines
                x_track[0][tail_length - 1] = x1
                x_track[1][tail_length - 1] = x2
                x_track[2][tail_length - 1] = x3
                x_track[3][tail_length - 1] = x4
                x_track[4][tail_length - 1] = x5
                x_track[5][tail_length - 1] = x6
                x_track[6][tail_length - 1] = x7
                x_track[7][tail_length - 1] = x8
                x_track[8][tail_length - 1] = x9
                x_track[9][tail_length - 1] = xi1
                x_track[10][tail_length - 1] = xi2
                x_track[11][tail_length - 1] = xi3
                x_track[12][tail_length - 1] = xi4
                # x_track[13][tail_length - 1] = sunmovex # place the sun position here

                y_track[0][tail_length - 1] = y1
                y_track[1][tail_length - 1] = y2
                y_track[2][tail_length - 1] = y3
                y_track[3][tail_length - 1] = y4
                y_track[4][tail_length - 1] = y5
                y_track[5][tail_length - 1] = y6
                y_track[6][tail_length - 1] = y7
                y_track[7][tail_length - 1] = y8
                y_track[8][tail_length - 1] = y9
                y_track[9][tail_length - 1] = yi1
                y_track[10][tail_length - 1] = yi2
                y_track[11][tail_length - 1] = yi3
                y_track[12][tail_length - 1] = yi4
                # y_track[13][tail_length - 1] = sunmovey # place sun position here

            if view == 0:
                # Iterates through the 2D list and draws the planet's trails
                for k in range(0, num_planets):
                    for j in range(1, tail_length - 1):
                        i = tail_length - j
                        if x_track[k][j - 1] != 0:
                            pygame.draw.line(
                                screen,
                                (255 - 255 * (i / tail_length),
                                255 - 255 * (i / tail_length),
                                255 - 255 * (i / tail_length)),
                                [x_track[k][j], y_track[k][j]],
                                [x_track[k][j - 1], y_track[k][j - 1]],
                                1)

                pygame.draw.circle(screen, (255, 255, 0), [x2, y2], 5)
                pygame.draw.circle(screen, (0, 255, 255), [x3, y3], 5)
                pygame.draw.circle(screen, (255, 255, 255), [x4, y4], 5)
                pygame.draw.circle(screen, (255, 0, 0), [x5, y5], 5)
                pygame.draw.circle(screen, (0, 255, 0), [x1, y1], 5)
                pygame.draw.circle(screen, (100, 50, 220), [x6, y6], 5)
                pygame.draw.circle(screen, (73, 155, 55), [x7, y7], 5)
                pygame.draw.circle(screen, (55, 75, 95), [x8, y8], 5)
                pygame.draw.circle(screen, (255, 102, 255), [x9, y9], 5)

                pygame.draw.circle(screen, (255, 255, 0), [xi2, yi2], 5)
                pygame.draw.circle(screen, (0, 255, 255), [xi3, yi3], 5)
                pygame.draw.circle(screen, (255, 255, 255), [xi4, yi4], 5)
                pygame.draw.circle(screen, (0, 255, 0), [xi1, yi1], 5)

            else:
                pygame.draw.circle(
                    screen, (255, 255, 0), [x2, int(scr_height / 2)], 5
                )
                pygame.draw.circle(
                    screen, (0, 255, 255), [x3, int(scr_height / 2)], 5
                )
                pygame.draw.circle(
                    screen, (255, 255, 255), [x4, int(scr_height / 2)], 5
                )
                pygame.draw.circle(
                    screen, (255, 0, 0), [x5, int(scr_height / 2)], 5
                )
                pygame.draw.circle(
                    screen, (0, 255, 0), [x1, int(scr_height / 2)], 5
                )
                pygame.draw.circle(
                    screen, (100, 50, 220), [x6, int(scr_height / 2)], 5
                )
                pygame.draw.circle(
                    screen, (73, 155, 55), [x7, int(scr_height / 2)], 5
                )
                pygame.draw.circle(
                    screen, (55, 75, 95), [x8, int(scr_height / 2)], 5
                )
                pygame.draw.circle(
                    screen, (255, 102, 255), [x9, int(scr_height / 2)], 5
                )

                pygame.draw.circle(screen, (255, 255, 0), [xi2, sun_i_y], 5)
                pygame.draw.circle(screen, (0, 255, 255), [xi3, sun_i_y], 5)
                pygame.draw.circle(screen, (255, 255, 255), [xi4, sun_i_y], 5)
                pygame.draw.circle(screen, (0, 255, 0), [xi1, sun_i_y], 5)

            # Updates the display with the new frame
            states = menu(
                screen,
                [pause, view, speed,0,0, click_now, input_active,
                 textbox_active, input_text],
                scr_width,
                scr_height
            )
            pause = states[0]
            speed = states[2]
            input_active = states[6]
            textbox_active = states[7]
            input_text = states[8]

            view = states[1]
            click_now = states[5]
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            pygame.display.flip()
            clock.tick(60)  # screen refresh rate
            if pause == 0:
                degree = degree + speed


def menu(screen, states, scr_width, scr_height):
    play_pause = [int(scr_width / 25), int(scr_height / 15 * 2.3)]
    toggle = [int(scr_width / 25), int(scr_height / 15 * 3.1)]
    adjust = [int(scr_width / 25), int(scr_height / 15 * 3.9)]
    reverse = [int(scr_width / 25), int(scr_height / 15 * 4.7)]
    upload = [int(scr_width / 25), int(scr_height / 15 * 5.5)]
    # [pause, view, speed, reverse, upload, click_now, input_active]
    pause = states[0]
    view = states[1]
    speed = states[2]
    rev = states[3]
    upl = states[4]
    click_now = states[5]
    input_active = states[6]
    textbox_active = states[7]
    input_text = states[8]

    action_flag, click_now, click_x, click_y = click_handler(screen, scr_width, scr_height, click_now)
    boxes(screen, scr_width, scr_height)
    menu_text(screen, scr_width, scr_height)
    text_handler(screen, "(" + str(speed) + "x)", int(scr_width / 4.2), int(scr_height / 15 * 3.9), 30, 120)

    if play_pause[0] + 200 > click_x > play_pause[0] and play_pause[1] + 30 > click_y > play_pause[1]:
        text_handler(screen, 'Pause/ Play', play_pause[0], play_pause[1], 30, 255)
        if action_flag == 1:
            pause = abs(pause - 1)

    elif toggle[0] + 200 > click_x > toggle[0] and toggle[1] + 30 > click_y > toggle[1]:
        text_handler(screen, 'Toggle View', toggle[0], toggle[1], 30, 255)
        if action_flag == 1:
            view = abs(view - 1)

    elif adjust[0] + 250 > click_x > adjust[0] and adjust[1] + 30 > click_y > adjust[1]:
        text_handler(screen, 'Adjust Speed', adjust[0], adjust[1], 30, 255)
        text_handler(screen, "(" + str(speed) + "x)", int(scr_width / 4.2), adjust[1], 30, 255)
        if action_flag == 1:
            speed = speed * 2
            if speed > 4:
                speed = 0.5
    elif reverse[0] + 300 > click_x > reverse[0] and reverse[1] + 30 > click_y > reverse[1]:
        text_handler(screen, 'Reverse Simulation', reverse[0], reverse[1], 30, 255)

    elif upload[0] + 200 > click_x > upload[0] and upload[1] + 30 > click_y > upload[1]:
        text_handler(screen, 'Upload Init File', upload[0], upload[1], 30, 255)
        if action_flag == 1:
            input_active = 1

    if input_active == 1:
        pause = 1
        prompt = "Please type the name or path of the init file:"
        pygame.draw.rect(screen,
                         (0, 0, 0),
                         pygame.Rect(
                             int(scr_width / 2.6),
                             int(scr_height / 2.7),
                             int(scr_width / 1.8),
                             int(scr_height / 5)
                         )
                         )
        pygame.draw.rect(screen,
                         (255, 255, 255),
                         pygame.Rect(
                             int(scr_width / 2.6),
                             int(scr_height / 2.7),
                             int(scr_width / 1.8),
                             int(scr_height / 5)
                         ),
                         2)
        pygame.draw.rect(screen,
                         (255, 255, 255),
                         pygame.Rect(
                             int(scr_width / 2.51),
                             int(scr_height / 2.2),
                             int(scr_width / 1.9),
                             int(scr_height / 15)
                         ),
                         2)
        text_handler(screen,
                     prompt,
                     int(scr_width / 2.6) + 10,
                     int(scr_height / 2.7) + 10,
                     25,
                     255)
        if int(scr_width / 2.51) + int(scr_width / 1.9) > click_x > int(scr_width / 2.51) and int(scr_height / 2.2) + \
                int(scr_height / 15) > click_y > int(scr_height / 2.2):
            if action_flag == 1:
                textbox_active = 1
            pygame.draw.rect(screen, (100, 100, 100),
                             pygame.Rect(int(scr_width / 2.51)+1, int(scr_height / 2.2)+1, int(scr_width / 1.9)-1,
                                         int(scr_height / 15)-1))
        if textbox_active == 1:
            pygame.draw.rect(screen, (100, 100, 100),
                             pygame.Rect(int(scr_width / 2.51) + 1, int(scr_height / 2.2) + 1, int(scr_width / 1.9) - 1,
                                         int(scr_height / 15) - 1))
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.KEYDOWN:
                    if input_active == 1:
                        if event.key == pygame.K_RETURN:
                            input_active = 0
                            textbox_active = 0
                            pause = 0
                        elif event.key == pygame.K_BACKSPACE:
                            input_text = input_text[:-1]
                        else:
                            input_text += event.unicode
            text_handler(screen, input_text + "|", int(scr_width / 2.51) + 3, int(scr_height / 2.17) + 3, 30, 255)
        else:
            text_handler(screen, input_text, int(scr_width / 2.51) + 3, int(scr_height / 2.17) + 3, 30, 255)
    return [pause, view, speed, reverse, upload, click_now, input_active, textbox_active, input_text]


# returns actionFlag, click_now, click_x, click_y
def click_handler(screen, scr_width, scr_height, click_now):
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()
    if click[0] == 1:
        if click_now == 0:
            return 1, 1, mouse[0], mouse[1]
        else:
            return 0, 1, mouse[0], mouse[1]
    else:
        return 0, 0, mouse[0], mouse[1]


def text_handler(screen, text, scr_x, scr_y, size, color):
    large_text = pygame.font.Font('freesansbold.ttf', size)
    text_surf = large_text.render(text, True, (color, color, color))
    text_rect = text_surf.get_rect()
    text_rect.topleft = (scr_x, scr_y)
    screen.blit(text_surf, text_rect)


def boxes(screen, scr_width, scr_height):
    pygame.draw.rect(screen,
                     (255, 255, 255),
                     pygame.Rect(
                         int(scr_width / 30),
                         int(scr_width / 30),
                         int(scr_width / 3.3),
                         int(scr_height / 2.4)
                     ),
                     2)
    pygame.draw.rect(screen,
                     (255, 255, 255),
                     pygame.Rect(
                         int(scr_width / 30),
                         int(scr_height / 2),
                         int(scr_width / 3.3),
                         int(scr_height / 2.2)
                     ),
                     2)
    pygame.draw.rect(screen,
                     (255, 255, 255),
                     pygame.Rect(
                         int(scr_width / 2.8),
                         int(scr_width / 30),
                         int(scr_width / 1.64),
                         int(scr_height / 1.1)
                     ),
                     2)


def menu_text(screen, scr_width, scr_height):
    text_handler(screen,
                 'Inner Planets',
                 int(scr_width / 13.8),
                 int(scr_height / 1.9),
                 35,
                 255)
    text_handler(screen,
                 'The Solar System',
                 int(scr_width / 2),
                 int(scr_height / 15),
                 35,
                 255)
    text_handler(screen,
                 'Menu',
                 int(scr_width / 25),
                 int(scr_height / 15),
                 50,
                 255)
    text_handler(screen,
                 'Pause/ Play',
                 int(scr_width / 25),
                 int(scr_height / 15 * 2.3),
                 30,
                 120)
    text_handler(screen,
                 'Toggle View',
                 int(scr_width / 25),
                 int(scr_height / 15 * 3.1),
                 30,
                 120)
    text_handler(screen,
                 'Adjust Speed',
                 int(scr_width / 25),
                 int(scr_height / 15 * 3.9),
                 30,
                 120)
    text_handler(screen,
                 'Reverse Simulation',
                 int(scr_width / 25),
                 int(scr_height / 15 * 4.7),
                 30,
                 120)
    text_handler(screen,
                 'Upload Init File',
                 int(scr_width / 25),
                 int(scr_height / 15 * 5.5),
                 30,
                 120)


if __name__ == "__main__":
    main()
