import pygame

SLIDER_COLOR = (255, 255, 255)
SLIDER_HANDLE_COLOR = (0, 0, 0)


ROUGE = (255, 0, 0)
BLEU = (0, 0, 255)
VERT = (0, 255, 0)

BLANC = (255, 255, 255)
GRIS = (127.5, 127.5, 127.5)
NOIR  = (0, 0, 0)
GRIS_CLAIR = (200, 200, 200)

SCROLL_WINDOW_WIDTH = 300
SCROLL_WINDOW_HEIGHT = 600
SLIDER_HEIGHT = 20
SLIDER_MARGIN = 30
NUM_SLIDERS = 1

class Slider:
    def __init__(self, x, y, width, height, min_value, max_value, initial_value, id):
        self.rect = pygame.Rect(x, y, width, height)
        self.width = width
        self.height = height
        self.min_value = min_value
        self.max_value = max_value
        self.value = initial_value
        self.handle_width = 10  # Augmenter la largeur du handle pour une meilleure manipulation
        self.handle_height = height  # Augmenter la hauteur du handle pour une meilleure manipulation
        self.handle_rect = pygame.Rect(x + (initial_value - min_value) / (max_value - min_value) * (width - self.handle_width), y, self.handle_width, self.handle_height)
        self.dragging = False
        self.id = id
        self.font = pygame.font.Font(None, 20)  # Police par défaut avec une taille de 20
        self.text_color = (0, 0, 0)  # Noir

    def draw(self, screen):
        # Dessiner la ligne du slider
        pygame.draw.rect(screen, SLIDER_COLOR, self.rect)
        # Dessiner le handle
        pygame.draw.rect(screen, SLIDER_HANDLE_COLOR, self.handle_rect)

        id_text = self.font.render(str(self.id), True, self.text_color)
        text_x = self.rect.x - id_text.get_width() - 10  # 10 pixels d'écart entre le texte et le slider
        text_y = self.rect.y + (self.rect.height - id_text.get_height()) // 2  # Centrer le texte verticalement
        screen.blit(id_text, (text_x, text_y))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.handle_rect.collidepoint(event.pos):
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                self.handle_rect.x = max(self.rect.x, min(event.pos[0] - self.handle_width // 2, self.rect.x + self.rect.width - self.handle_width))
                self.value = self.min_value + (self.handle_rect.x - self.rect.x) / (self.rect.width - self.handle_width) * (self.max_value - self.min_value)
                return self.value
        return None
    
class ScrollableWindow:
    def __init__(self, x, y, width, height, sliders):
        self.rect = pygame.Rect(x, y, width, height)
        self.sliders = sliders
        self.scroll_y = 0
        self.drag_start_y = None


    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.drag_start_y = event.pos[1]  # Stocker la position de départ du clic
        elif event.type == pygame.MOUSEBUTTONUP:
            self.drag_start_y = None  # Réinitialiser lorsque le clic est relâché
        elif event.type == pygame.MOUSEMOTION:
            if self.drag_start_y is not None:
                # Calculer le déplacement vertical
                delta_y = event.pos[1] - self.drag_start_y
                self.scroll_y -= delta_y  # Modifier la position de défilement en fonction du déplacement
                self.drag_start_y = event.pos[1]  # Mettre à jour la position de départ du clic

        # Optionnel : Gérer le défilement avec la molette de la souris ou trackpad
        elif event.type == pygame.MOUSEWHEEL:
            self.scroll_y += event.y * 10  # Ajuster la vitesse de défilement en fonction de la molette


    def draw(self, screen):
        # Dessiner la fenêtre de défilement
        pygame.draw.rect(screen, GRIS_CLAIR, self.rect)
        
        # Dessiner les sliders
        for i, slider in enumerate(self.sliders):
            y = self.rect.y + SLIDER_MARGIN + i * (SLIDER_HEIGHT + SLIDER_MARGIN) - self.scroll_y
            if y + SLIDER_HEIGHT > self.rect.y and y < self.rect.bottom:
                slider.rect.y = y
                slider.handle_rect.y = y
                slider.draw(screen)

    def update(self):
        # Mettre à jour le défilement pour éviter que les sliders sortent de la fenêtre
        max_scroll = max(0, (len(self.sliders) * (SLIDER_HEIGHT + SLIDER_MARGIN)) - SCROLL_WINDOW_HEIGHT)
        self.scroll_y = max(0, min(self.scroll_y, max_scroll))