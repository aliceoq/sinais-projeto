import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as plt_patches
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter, sobel
import time

# Inicialização das constantes
MIN_RADIUS = 1
MAX_RADIUS = 151
EDGE_LIMIT = 0.005  # Limite do contorno
ROUND_OUTPUT = 3  # Formatação do valor final a ser mostrado
MAX_RECOMMENDED_DENSITY = 0.25

# Valores setados baseados em testes
COROA_WIDTH = [1, 3, 5, 7.5, 10]
INNER_CIRCLE_SCALE = 1.1

# Configuração da página
st.set_page_config("Projeto de sinais", "⚽")
st.header("Reconhecer bolas esportivas em uma imagem")
st.subheader(
    "Considere analisar imagens com círculos de tamanho considerável, pois iremos detectar a bola com maior sinal normalizado, baseado na matriz de convolução"
)
st.write(
    f"Tamanho mínimo do círculo = {MIN_RADIUS} px ; ",
    f"Tamanho máximo do círculo = {MAX_RADIUS - 1} px",
)


def display_image(image, label=""):
    try:
        st.image(image, label, clamp=True)
    except:
        st.error(f"Não foi possível disponibilizar a imagem de label: '{label}'")


def detect_edges(image, limit):
    # Aplicação do filtro Sobel na imagem para detecção de contornos
    image = sobel(image, 0) ** 2 + sobel(image, 1) ** 2
    image -= image.min()

    # tornar a imagem binária
    image = image > (image.max() * limit)
    image.dtype = np.int8

    return image


# Cria uma coroa circular baseada no raio interno e externo
def make_coroa_kernel(outer_radius, coroa_width):
    # outer_radius - raio maximo do filtro
    # inner_radius - raio do circulo de dentro do filtro
    # Criar template da coroa
    grids = np.mgrid[-outer_radius : outer_radius + 1, -outer_radius : outer_radius + 1]
    kernel_template = grids[0] ** 2 + grids[1] ** 2

    # Círculos em valores booleanos
    outer_circle = kernel_template <= (outer_radius ** 2)
    inner_circle = kernel_template < ((outer_radius - coroa_width) ** 2)

    # Transforma os valores para inteiro
    outer_circle.dtype = inner_circle.dtype = np.int8
    inner_circle = inner_circle * INNER_CIRCLE_SCALE
    coroa = outer_circle - inner_circle

    return coroa


@st.cache(suppress_st_warning=True, show_spinner=False)
def detect_balls(image, list_of_radius, coroa_width):
    # Realizamos uma convolução FFT (convolução após a transformada) em todos os raios possíveis do array,
    # considerando a largura da coroa dada.

    # Quanto menor o tamanho da coroa -> mais preciso será a transformada
    convolution_matrix = np.zeros((list_of_radius.size, image.shape[0], image.shape[1]))

    # matriz de convolução == kernel
    for index, rad in enumerate(list_of_radius):
        kernel = make_coroa_kernel(rad, coroa_width)
        convolution_matrix[index, :, :] = fftconvolve(
            image, kernel, "same"
        )  # same -> o output terá mesmo tamanho do parametro 1

    return convolution_matrix


# Encontrar o círculo (bola) com maior sinal normalizado da imagem
@st.cache(suppress_st_warning=True, show_spinner=False)
def top_circle(convolution_matrix, list_of_radius):
    # identificar os cículos com maiores sinais
    maxima = []
    max_positions = []
    max_signal, final_radius, (circle_y, circle_x) = 0, 0, (0, 0)

    for index, radius in enumerate(list_of_radius):
        max_positions.append(
            np.unravel_index(
                convolution_matrix[index].argmax(), convolution_matrix[index].shape
            )
        )
        maxima.append(convolution_matrix[index].max())

        # usa o raio para normalizar o sinal
        signal = maxima[index] / np.sqrt(float(radius))

        if signal > max_signal:
            max_signal = signal
            (circle_y, circle_x) = max_positions[index]
            final_radius = radius

    return ((circle_x, circle_y), final_radius)


def run(image):
    if image.ndim > 2:
        image = np.mean(image, axis=2)

    # Uso do filtro gaussiano
    image = gaussian_filter(image, 2)

    # Definir contornos da imagem e densidade dos sinais
    edges = detect_edges(image, EDGE_LIMIT)
    edge_list = np.array(edges.nonzero())
    density = float(edge_list[0].size) / edges.size
    st.info(f"Densidade do sinal da imagem: {density}")
    if density > MAX_RECOMMENDED_DENSITY:
        st.warning(
            f"A densidade do sinal da imagem é muito grande (maior que {str(MAX_RECOMMENDED_DENSITY).replace('.', ',')}). Isso pode afetar consideravelmente na precisão do resultado!"
        )

    # criar kernels e detectar círculos
    # aqui, não vamos incluir o raio = 151, vai até 150
    list_of_radius = np.arange(MIN_RADIUS, MAX_RADIUS)
    for width in COROA_WIDTH:
        with st.spinner(
            f"Calculando círculo com maior sinal para largura da coroa circular = {width} px"
        ):
            convolution_matrix = detect_balls(edges, list_of_radius, width)
            center, radius = top_circle(convolution_matrix, list_of_radius)
        st.success(
            f"Círculo detectado no ponto {center}, com raio = {radius} px e largura da coroa circular = {width} px"
        )
        display_results(image, edges, center, radius)


def display_results(image, edges, center, radius):
    try:
        # Imagem com o contorno
        plt.gray()
        fig = plt.figure(1)
        fig.clf()
        subplots = []
        subplots.append(fig.add_subplot(1, 2, 1))
        plt.imshow(edges)
        plt.title("Contornos da imagem")

        # Imagem com o resultado
        subplots.append(fig.add_subplot(1, 2, 2))
        plt.imshow(image)
        plt.title(f"Centro: {str(center)}; Raio: {radius} px")

        # Desenhar o círculo detectado
        blob_circ = plt_patches.Circle(center, radius, fill=False, ec="red")
        plt.gca().add_patch(blob_circ)
        plt.axis("image")

        fig
    except:
        st.error(f"Não foi possível plotar as imagens do resultado.")


def main():
    st.warning("Anexar imagens muito grandes pode afetar o tempo de execução!")
    images = st.file_uploader(
        "Faça o upload de uma imagem ou mais imagens a ser(em) testada(s)",
        ["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )

    if st.button("Rodar o programa!"):
        if images:
            with st.sidebar:
                st.write("Progresso")
                progress_bar = st.progress(0)
                if len(images) > 3:
                    st.warning(
                        "Analisar muitos arquivos pode deixar o programa consideravelmente mais lento!"
                    )
            count = 1
            start_time = time.time()
            for image in images:
                st.markdown(f"## Teste {count}")
                try:
                    display_image(image, "Imagem original")
                    image = plt.imread(image)
                    run(image)
                except:
                    st.error(
                        f"Não foi possível rodar o programa para o teste {count}, tente novamente."
                    )
                finally:
                    progress_bar.progress((count) / len(images))
                    count += 1
            with st.sidebar:
                end_time = time.time()
                full_time = end_time - start_time
                minutes = full_time // 60
                if minutes:
                    seconds = full_time - (minutes * 60)
                    st.markdown(
                        f"#### Tempo de execução: {int(minutes)} {'minutos' if minutes > 1 else 'minuto'} e {int(seconds)} {'segundos' if seconds > 1 else 'segundo'} "
                    )
                else:
                    st.markdown(f"#### Tempo de execução: {int(full_time)} segundos")
            st.balloons()  # Finished
        else:
            st.warning(
                "Parece que você não selecionou nenhuma imagem, tente novamente!"
            )
            st.stop()


main()
