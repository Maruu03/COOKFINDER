# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settings
import helper
import openai
import cv2
import time
from ultralytics import YOLO
from openai import OpenAI

#variables locales
from dotenv import load_dotenv
import os
import time

load_dotenv()

#nombreRecetas = []
#count = 0
# Función para detectar objetos
def detect_objects(image_path):
    # Cargar el modelo YOLOv5 preentrenado
    model = YOLO('/Users/isabella/Library/CloudStorage/OneDrive-EscuelaSuperiorPolitécnicadelLitoral/ESPOL/PAOII-2023/TAWS/PRETAWS/weights/prueba1.pt')
    # Ejecutar inferencia en la imagen de entrada
    results = model(image_path)
    # Procesar los resultados de la detección
    detected_objects = []
    for result in results:
        for box in result.boxes:
            class_id = result.names[box.cls[0].item()]
            conf = round(box.conf[0].item(), 2)
            detected_objects.append((class_id, conf))
    return detected_objects

# Función para buscar recetas
def buscarRecetas(ingredientes_busqueda, recetas):
    if len(ingredientes_busqueda) > 0:
        coincidencias_por_receta = []
        
        for receta, detalles in recetas.items():
            ingredientes_receta = detalles['ingredientes']
            foto = detalles['foto']
            coincidencias = len(set(ingredientes_busqueda) & set(ingredientes_receta))
            proporcion_coincidencias = coincidencias / len(ingredientes_receta) if len(ingredientes_receta) > 0 else 0
            
            coincidencias_por_receta.append((receta, proporcion_coincidencias,foto))
        
        coincidencias_por_receta.sort(key=lambda x: x[1], reverse=True)
        primeras_tres_recetas = coincidencias_por_receta[:3]

        nombres_recetas = [receta[0] for receta in primeras_tres_recetas]
        fotos_recetas = [receta[2] for receta in primeras_tres_recetas]

        return primeras_tres_recetas, nombres_recetas, fotos_recetas
    
    return None, None
    #return [receta for receta, _ in coincidencias_por_receta[:3]]


def crearMensaje(text, nombre, url):
    try:
        client = OpenAI(
            # This is the default and can be omitted
            api_key = os.getenv("OPENAI_API_KEY")

        )
        
        response = client.chat.completions.create(
                messages=[
                    {"role": "system", 
                     "content": f"Eres un chef que da las instrucciones para hacer la receta que se indica. Primero dame una lista enumerada de ingredientes especificos que necesito para hacer la receta (incluye los {text} (si los ingredientes estan en ingles traduce a español)). Luego, enumera los pasos para la receta, no agregues ningún otro texto (no recomendaciones ni comentarios extras). Recuerda que en la lista de ingredientes y en la receta debes obligatoriamente utilizar los ingredientes de {text}."},
                    {"role": "user", 
                     "content": nombre},
                ],
                model="gpt-3.5-turbo",
            )

        st.info(nombre)
        instructions = response['choices'][0]['message']['content']

        for i, instruction in enumerate(instructions, start=1):
            st.write(f"{instruction.strip()}")
        st.image(url)
        return False
    
    except Exception as ex:
        print(ex)
        with st.spinner("Por favor espere unos segundos..."):
            time.sleep(20)
        print(ex)
        return True

def imprimirInstrucciones(run, objetos_detectados, nombre, url):
    text = ""
    for string in objetos_detectados:
        if string.capitalize() == "Verde":
            string = "Platano Verde"
        if not(string in text):
            text = text + string + " "
        
    print(text)
    repetir = True
    count = 0
    while repetir:
        repetir = crearMensaje(text,nombre, url)

def display_tracker_options():
    #display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True
    #if display_tracker == 'Yes'
    if is_display_tracker:
        #tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        tracker_type = "bytetrack.yaml"
        return is_display_tracker, tracker_type
    return is_display_tracker, None

def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
    None
    """

    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)

    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Video detectado',
                   channels="BGR",
                   use_column_width=True
                   )

def capturar(conf, model, st_frame, image,recetas, class_labels, is_display_tracking=None, tracker=None):
    
    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)

    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Video detectado',
                   channels="BGR",
                   use_column_width=True
                   )
    boxes = res[0].boxes
    objetos_detectados = []
    if boxes:  # Verificamos si hay cajas detectadas
        for box in boxes:
            class_index = box.cls.item()  # Get the class index of the box
            label_name = class_labels[class_index] 
            if not(label_name.capitalize() in objetos_detectados):
                objetos_detectados.append(label_name.capitalize())
    objetos = objetos_detectados
    best, names, fotos = buscarRecetas(objetos_detectados, recetas)
    poner = True
    return objetos, best, names, poner, fotos



# Diccionario de recetas

recetas = {
    'Arroz con Menestra y Carne': { 'ingredientes': ['Verde','Carne', 'Cebolla'], 'foto': "images/menestra.webp" },
    'Ensalada de Papa': { 'ingredientes': ['Potato', 'Tomato', 'Cebolla'], 'foto': "images/ensaladaPapa.jpeg" },
    'Pure con Carne': { 'ingredientes': ['Potato', 'Carne'], 'foto': "images/pure.jpeg" },
    'Bistec de Carne': { 'ingredientes': ['Verde','Carne', 'Cebolla'], 'foto': "images/bistec.jpeg" },
    'Tortilla de Papa': { 'ingredientes': ['Potato', 'Carne', 'Cebolla'], 'foto': "images/tortilla.webp" },
    'Empanadas de Verde': { 'ingredientes': ['Verde','Carne', 'Cebolla'], 'foto': "images/empanadas.jpeg" },
    'Seco de Pollo': { 'ingredientes': ['Verde','Pollo', 'Cebolla', 'Tomato'], 'foto': "images/seco.jpeg" },
    'Pollo Guisado': { 'ingredientes': ['Tomato','Pollo', 'Cebolla'], 'foto': "images/guisado.jpeg" },
    'Sopa de Pollo': { 'ingredientes': ['Pollo','Potato', 'Cebolla'], 'foto': "images/sopa.webp" },
    
    'Llapingacho': { 'ingredientes': ['Potato', 'Cebolla', 'Tomato'], 'foto': "images/llapi.webp" },
    'Estofado de Pollo': { 'ingredientes': ['Potato', 'Cebolla', 'Pollo'], 'foto': "images/estofado.jpeg" },
    'Churrasco': { 'ingredientes': ['Carne', 'Cebolla', 'Tomato'], 'foto': "images/chu.jpeg" },
    'Lomo Saltado': { 'ingredientes': ['Carne', 'Cebolla', 'Tomato', 'Potato'], 'foto': "images/lomo.jpeg" },
    'Papas Rellenas de Carne': { 'ingredientes': ['Carne', 'Cebolla', 'Tomato', 'Potato'], 'foto': "images/rellenas.jpeg" },
    'Arroz con Pollo': { 'ingredientes': ['Pollo', 'Cebolla', 'Tomato'], 'foto': "images/arrozPollo.jpeg" },
    'Tigrillo': { 'ingredientes': ['Verde', 'Cebolla'], 'foto': "images/tigrillo.webp" }
}


# Setting page layout
st.set_page_config(
    page_title="Clasificación de ingredientes utilizando YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.markdown(
    """
    <div style="text-align: center;">
        <h1>COOKFINDER</h1>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
        body {
            background-color: #D2691E;
        }
    </style>
    """,
    unsafe_allow_html=True
)
#st.title("CookFinder")

# Sidebar
#st.sidebar.header("Configuración del modelo")

# Model Options
#model_type = st.sidebar.radio(
    #"Escoja una tarea", ['Detection'])

# Configurar el color de fondo del sidebar
st.markdown(
    """
    <style>
        .sidebar {
            background-color: ##FFAC11;  /* Cambia este valor al color que desees */
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.image("images/logo.png", use_column_width=True)
# Selecting Detection
model_path = Path('avance.pt')

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"No se pudo cargar el modelo. Verifique el path indicado: {model_path}")
    st.error(ex)
# Initialize confidence (example value, adjust as needed)
confidence = 0.1
class_labels = model.names
st.sidebar.header("¿Cómo desea detectar sus ingredientes?")
source_radio = st.sidebar.radio(
    "Escoja una fuente", settings.SOURCES_LIST)

source_img = None
# If image is selected
container_style = """
    padding: 10px;
    border: 2px solid #ff5733; /* Border color: change this hex color code as needed */
    border-radius: 5px; /* Optional: adds rounded corners */
"""

if source_radio == settings.IMAGE:
    st.empty()
    poner1 = False
    best1 = []
    names1 = []
    objetos1 = []
    source_img = st.sidebar.file_uploader(
        "Escoja una imagen...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Imagen por default",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Imagen subida",
                         use_column_width=True)
        except Exception as ex:
            st.error("Ocurrió un error al abrir la imagen")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Imagen detectada',
                     use_column_width=True)
        else:
            objetos_detectados1 = []
            if st.sidebar.button('Objetos detectados'):
                res = model.predict(uploaded_image,
                                    conf=confidence
                                    )
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Imagen detectada',
                         use_column_width=True)

                if boxes:  # Verificamos si hay cajas detectadas
                    for box in boxes:
                        class_index = box.cls.item()  # Get the class index of the box
                        label_name = class_labels[class_index] 
                        if not(label_name.capitalize() in objetos_detectados1):
                            objetos_detectados1.append(label_name.capitalize())
                objetos1 = objetos_detectados1
                best1, names1, fotos1 = buscarRecetas(objetos_detectados1, recetas)
                poner1 = True
                #receta_seleccionada = st.radio('Seleccione una receta:', nombres)
    
                #if receta_seleccionada:
                    #num = mejoresRecetas.index(receta_seleccionada) + 1
                    #imprimirInstrucciones(num, objetos_detectados)
                
                #try:
                   # with st.expander("Resultados de deteccción"):
                      # for box in boxes:
                          #  st.write(box.data)
                #except Exception as ex:
                    # st.write(ex)
                    #st.write("¡No se ha subido ninguna imagen todavía!")
    if poner1 and (len(objetos1)!=0):
        st.divider()
        st.markdown("<h3 style='text-align: center;'>Las tres mejores recetas</h3>", unsafe_allow_html=True)
        i = 1
        for re1,fo1 in zip(names1,fotos1):
            imprimirInstrucciones(i,objetos1,re1,fo1)
            i = i + 1
            st.divider()
            
        
    elif poner1 and (len(objetos1)== 0):
        st.info("No se detectaron objetos")
    
        
    
elif source_radio == settings.VIDEO:
    #helper.play_stored_video(confidence, model)
    source_vid = st.sidebar.selectbox(
        "Escoja un video...", settings.VIDEOS_DICT.keys())

    is_display_tracker, tracker = display_tracker_options()

    poner2 = False
    best2 = []
    names2 = []
    objetos2 = []

    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button('Detectar los objetos del video'):
        
        try:
            vid_cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            i = 0
            while (i!=30):
                success, image = vid_cap.read()
                i = i + 1
                if success:
                    objetos2, best2, names2, poner2, fotos2 = capturar(confidence,
                                             model,
                                             st_frame,
                                             image, recetas, class_labels,
                                             is_display_tracker,
                                             tracker
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
    
    if poner2 and (len(objetos2) != 0):
        st.divider()
        st.markdown("<h3 style='text-align: center;'>Las tres mejores recetas</h3>", unsafe_allow_html=True)
        i = 1
        for objeto in objetos2:
            print(objeto)
        for tupla in best2:
            elem1 = str(tupla[0])
            elem2 = str(tupla[1])
            print(elem1 + " " + elem2)
        for name in names2:
            print(name)
        for re2,fo2 in zip(names2,fotos2):
            imprimirInstrucciones(i,objetos2,re2,fo2)
            i = i + 1
            st.divider()
            
    elif poner2 and (len(objetos2)== 0):
        st.info("No se detectaron objetos")
    

elif source_radio == settings.WEBCAM:
    st.empty()
    is_display_tracker, tracker = display_tracker_options()
    poner3 = False
    best3 = []
    names3 = []
    objetos3 = []
    # Inicializar la webcam
    video_stream = cv2.VideoCapture(0)
    ret, frame = video_stream.read()
    colu1, colu2 = st.columns([2,1])

    if not ret:
        video_stream = cv2.VideoCapture(1)

    if not video_stream.isOpened():
        st.error("No se pudo abrir la cámara web.")

    st_frame = st.empty()
    capture_button = st.sidebar.button("Capturar")

   
    with colu1:
        while not capture_button:
            ret, frame = video_stream.read()

            if not ret:
                st.error("Error al capturar el fotograma de la webcam.")
                break

            if frame is not None:
                # Detección de ingredientes en el fotograma actual
                _display_detected_frames(confidence, model, st_frame, frame, is_display_tracker, tracker)

            
    with colu2:
        ret, frame = video_stream.read()
        if capture_button:
            objetos3, best3, names3, poner3, fotos3 = capturar(confidence, model, st_frame, frame, recetas, class_labels, is_display_tracker, tracker)

    if poner3 and (len(objetos3) != 0):
        st.divider()
        st.markdown("<h3 style='text-align: center;'>Las tres mejores recetas</h3>", unsafe_allow_html=True)
        i = 1
        for re3,fo3 in zip(names3,fotos3):
            imprimirInstrucciones(i,objetos3,re3, fo3)
            i = i + 1
            st.divider()
            
    elif poner3 and (len(objetos3)== 0):
        st.info("No se detectaron objetos")
    # Liberar recursos
    video_stream.release()

else:
    st.error("¡Seleccione un tipo de fuente válido!")