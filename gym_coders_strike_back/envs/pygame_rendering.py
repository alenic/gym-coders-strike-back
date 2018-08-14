from pygame.locals import *
import pygame
import sys
import numpy as np

class Viewer(object):
  def __init__(self, width, height, display=None):
    pygame.init()
    
    self.width = width
    self.height = height

    self.screen = pygame.display.set_mode((width, height), 0, 32)
    surface = pygame.Surface(self.screen.get_size())
    self.surface = surface.convert()

    self.fpsClock=pygame.time.Clock()

    self.pods = []
    self.checkpoints = []
    self.backgroundImage = None

    self.isOpen = True

  def addPod(self, pod):
    self.pods.append(pod)
  
  def addCheckpoint(self, checkpoint):
    self.checkpoints.append(checkpoint)
  
  def setBackground(self, imgPath):
    self.backgroundImage = pygame.image.load(imgPath)
  
  def render(self):
    for event in pygame.event.get():
      if event.type == QUIT:
        self.isOpen = False
        pygame.quit()
        return False

    if self.backgroundImage is not None:
      self.surface.blit(self.backgroundImage, (0, 0))
    else:
      self.surface.fill((0,0,0))

    for ckpt in self.checkpoints:
      if ckpt.visible:
        cx, cy = ckpt.getCoordinates()
        self.surface.blit(ckpt.image, (cx, cy))
        font = pygame.font.Font(None, 24)
        text = font.render(str(ckpt.number), 1, (10, 255, 10))
        textpos = text.get_rect()
        textpos.centerx = ckpt.pos[0]
        textpos.centery = ckpt.pos[1]+1
        self.surface.blit(text, textpos)

    for pod in self.pods:
      cx, cy = pod.getCoordinates()
      self.surface.blit(pod.image, (cx, cy))
  
    self.screen.blit(self.surface, (0,0))
    pygame.display.flip()
    pygame.display.update()

    return self.isOpen
  
  def close(self):
    pygame.quit()
    
  
class Geometry(object):
  def __init__(self, imgPath, pos=(0,0), width=128, height=128):
    self.image = pygame.image.load(imgPath)
    self.image = pygame.transform.scale(self.image, (int(width),int(height)))
    self.pos = pos
    self.width = width
    self.height = height
    self.visible = True

  def setPos(self, pos):
    self.pos = pos
  
  def setVisible(self, value):
    self.visible = value
  
  def getCoordinates(self):
    return self.pos[0]- self.width//2, self.pos[1] - self.height//2


class Pod(Geometry):
  def __init__(self, imgPath, pos=(0,0), theta=0.0, width=128, height=128):
    Geometry.__init__(self, imgPath, pos, width, height)
    self.theta = theta
    self.baseImage = self.image
    self.image = pygame.transform.rotate(self.baseImage, self.theta*180.0/np.pi)
  
  def rotate(self, angle):
    self.theta = angle
    self.image = pygame.transform.rotate(self.baseImage, -angle*180.0/np.pi)


class Checkpoint(Geometry):
  def __init__(self, imgPath,pos=(0,0), number=0, width=128, height=128):
    Geometry.__init__(self, imgPath, pos, width, height)
    self.number = number
  
  def setNumber(self, number):
    self.number = number