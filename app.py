import streamlit as st
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as plt_patches
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter, sobel

# inicialização das constantes
MIN_RADIUS = 5
MAX_RADIUS = 205
RADIUS_STEP = 5  # diferença entre cada raio (no array de raios)
COROA_WIDTH = 5
EDGE_LIMIT = 0.005
NEG_INTERIOR_WEIGHT = 1.1

st.title('Projeto de sinais')
st.header('Reconhecer bolas esportivas em uma imagem')
st.subheader('Considere analisar imagens com círculos de tamanho considerável, pois iremos detectar o maior na imagem')
st.write(f"Tamanho mínimo do círculo = {MIN_RADIUS} px ; ", f"Tamanho máximo do círculo = {MAX_RADIUS} px")

def detect_edges(image, limit):
    # Aplicação do filtro Sobel na imagem para detecção de contornos
    image = sobel(image, 0)**2 + sobel(image, 1)**2 
    image -= image.min()
    
    # tornar a imagem binária
    image = image > (image.max() * limit)
    image.dtype = np.int8
        
    return image

# NÃO FAÇO IDEIA DO QUE SEJA ISSO
def make_coroa_kernel(outer_radius, coroa_width):
    # Cria uma coroa circular baseada no raio interno e externo
    
    grids = np.mgrid[-outer_radius : outer_radius+1, -outer_radius : outer_radius+1]
    
    kernel_template = grids[0]**2 + grids[1]**2
    
    outer_circle = kernel_template <= outer_radius**2
    inner_circle = kernel_template < (outer_radius - coroa_width)**2
    
    # Transforma os valores para inteiro
    outer_circle.dtype = inner_circle.dtype = np.int8
    inner_circle = inner_circle * NEG_INTERIOR_WEIGHT
    coroa = outer_circle - inner_circle
    return coroa

def detect_circles(image, list_of_radius, coroa_width):
    # Realizamos uma convolução FFT em todos os raios possíveis do array,
    # considerando a largura da coroa dada.
    
    # Quanto menor o tamanho da coroa -> mais preciso será a transformada
    convolution_matrix = np.zeros((list_of_radius.size, image.shape[0], image.shape[1]))

    # matriz de convolução == kernel
    for index, rad in enumerate(list_of_radius):
        kernel = make_coroa_kernel(rad, coroa_width)
        convolution_matrix[index, :, :] = fftconvolve(image, kernel, 'same')

    return convolution_matrix

def top_circles(convolution_matrix, list_of_radius):
    # identificar os cículos com maiores sinais
    maxima = []
    max_positions = []
    max_signal, final_radius, (circle_y, circle_x) = 0, 0, (0, 0)
    
    for index, radius in enumerate(list_of_radius):
        max_positions.append(np.unravel_index(convolution_matrix[index].argmax(), convolution_matrix[index].shape))
        maxima.append(convolution_matrix[index].max())
        
        # usa o raio para normalizar
        signal = maxima[index]/np.sqrt(float(radius))
        
        if signal > max_signal:
            max_signal = signal
            (circle_y, circle_x) = max_positions[index]
            final_radius = radius
        st.info(f"Valor máximo do sinal (raio = {radius} px): {maxima[index]} {max_positions[index]}, sinal normalizado: {signal}")
        
    
    return (circle_x, circle_y), final_radius #final_radius -> list_of_radius[max_index]

def run(image):
    if image.ndim > 2:
        image = np.mean(image, axis=2)
    
    # Uso do filtro gaussiano
    image = gaussian_filter(image, 2)

    # Definir contornos da imagem e densidade dos sinais
    edges = detect_edges(image, EDGE_LIMIT)
    edge_list = np.array(edges.nonzero())
    density = float(edge_list[0].size) / edges.size
    st.info(f"Densidade do sinal: {density}")
    if density > 0.25:
        st.warning("A densidade do sinal da imagem é muito grande (maior que 0,25), isso pode afetar consideravelmente na precisão do resultado")
    
    # criar kernels e detectar círculos
    # aqui, não vamos incluir o raio = 205, vai até 200
    list_of_radius = np.arange(MIN_RADIUS, MAX_RADIUS, RADIUS_STEP)
    convolution_matrix = detect_circles(edges, list_of_radius, COROA_WIDTH)
    center, radius = top_circles(convolution_matrix, list_of_radius)
    st.success(f"Círculo detectado no ponto {center}, com raio = {radius} px")
    display_results(image, edges, center, radius, "resultado.png")


def display_results(image, edges, center, radius, output=None): 
    # Imagem com o resultado
    plt.gray()
    fig = plt.figure(1)
    fig.clf()
    subplots = []
    subplots.append(fig.add_subplot(1, 2, 1))
    plt.imshow(edges)
    plt.title('Contornos da imagem')
    
    # Imagem original
    subplots.append(fig.add_subplot(1, 2, 2))
    plt.imshow(image)
    plt.title(f"Ponto: {str(center)}, Raio: {radius} px")
    
    # Desenhar o círculo detectado
    blob_circ = plt_patches.Circle(center, radius, fill=False, ec='red')
    plt.gca().add_patch(blob_circ)
    plt.axis('image')
        
    try:
        st.pyplot(fig)
        st.balloons()
    except Exception as err:
        st.error(f"Não foi possível plotar a imagem.")
        st.exception(err)
        
    
def main():
    image = st.file_uploader("Faça o upload da imagem a ser testada", ['jpg', 'png', 'webp'])
    if image:
        image = plt.imread(image)
        run(image)

main()