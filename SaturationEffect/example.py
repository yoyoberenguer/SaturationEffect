"""
EXAMPLE
"""
import time

try:
    import pygame
    from pygame.surfarray import array3d, pixels3d, pixels_alpha
except ImportError:
    raise ImportError('\n<pygame> library is missing on your system.'
                      "\nTry: \n   C:\\pip install pygame on a window command prompt.")

from SaturationEffect import saturation24

pygame.init()
screen = pygame.display.set_mode((640, 480))

image = pygame.image.load('../Assets/p1.png').convert_alpha()
image = pygame.transform.smoothscale(image, (640, 480))

sat = 1.0
v = 0.01
timer = time.time()
CLOCK = pygame.time.Clock()
while 1:
    pygame.display.set_caption(
        "Saturation effect using saturation24 saturation = %s" % round(sat, 2))
    pygame.event.pump()
    screen.fill((0, 0, 0))

    sat -= v
    if sat < -1.0:
        sat = -1.0
        v *= -1
    elif sat > 1.0:
        sat = 1.0
        v *= -1

    saturated_image = saturation24(pixels3d(image), sat)
    screen.blit(saturated_image, (0, 0))
    CLOCK.tick(30)

    if time.time() - timer > 10:
        break

    pygame.display.flip()

pygame.quit()
print('Have a nice day')