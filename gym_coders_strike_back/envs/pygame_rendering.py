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

    self.backgroundImage = None
    self.pods = []
    self.checkpoints = []
    self.text = None

    self.isOpen = True

  def addPod(self, pod):
    self.pods.append(pod)
  
  def addCheckpoint(self, checkpoint):
    self.checkpoints.append(checkpoint)

  def addText(self, text):
    if self.text is None:
      self.text = text

  def removeText(self):
    if self.text is not None:
      self.text = None
  
  def setBackground(self, imgPath):
    self.backgroundImage = pygame.image.load(imgPath)


  def render(self):
    for event in pygame.event.get():
      if event.type == QUIT:
        self.isOpen = False
        pygame.quit()
        return False

    # Background
    if self.backgroundImage is not None:
      self.surface.blit(self.backgroundImage, (0, 0))
    else:
      self.surface.fill((0,0,0))

    # Checkpoints
    for ckpt in self.checkpoints:
      if ckpt.visible:
        cx, cy = ckpt.getCoordinates()
        self.surface.blit(ckpt.image, (cx, cy))
        font = pygame.font.Font(None, 24)
        text = font.render(str(ckpt.number), 1, (255, 255, 255))
        textpos = text.get_rect()
        textpos.centerx = ckpt.pos[0]
        textpos.centery = ckpt.pos[1]+1
        self.surface.blit(text, textpos)

    # Pods
    for pod in self.pods:
      cx, cy = pod.getCoordinates()
      self.surface.blit(pod.image, (cx, cy))
      #if pod.target_arrow is not None:
      #  pygame.draw.line(self.surface, GREEN, pod.pos, pod.target_arrow, width=1)

    # Text
    if self.text is not None:
      font = pygame.font.Font(None, self.text.fontSize)
      text = font.render(self.text.text, 1, self.text.color, self.text.backgroundColor)
      textpos = text.get_rect()
      textpos.left = self.text.pos[0]
      textpos.top = self.text.pos[1]
      self.surface.blit(text, textpos)

    self.screen.blit(self.surface, (0,0))
    pygame.display.flip()
    pygame.display.update()

    pygame.time.wait(50)

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
    self.target_pos = None
  
  def rotate(self, angle):
    self.theta = angle
    self.image = pygame.transform.rotate(self.baseImage, -angle*180.0/np.pi)
  
  def setTargetPos(self, target_pos):
    self.target_pos = target_pos


class Checkpoint(Geometry):
  def __init__(self, imgPath,pos=(0,0), number=0, width=200, height=200):
    Geometry.__init__(self, imgPath, pos, width, height)
    self.number = number
  
  def setNumber(self, number):
    self.number = number

class Text(object):
  def __init__(self, text='text', pos=(0,0), color=(255,0,0), backgroundColor=None, fontSize=32):
    self.text = text
    self.pos = pos
    self.color = color
    self.fontSize = fontSize
    self.backgroundColor = backgroundColor
  
  def setText(self, text):
    self.text = text
