import streamlit
assert streamlit.__version__ >= "1.38.0"

from streamlit_drawable_canvas import st_canvas

import torch
assert torch.__version__ >= "2.0.0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import cv2

import torchvision.transforms as transforms

# import mlflow
# import mlflow.pytorch
# assert mlflow.__version__ >= "1.0.0"
# mlflow.set_tracking_uri("http://127.0.0.1:8081/")


streamlit.title('My Digit Recognizer')
streamlit.markdown('''
Try to write a digit!
''')

SIZE=192
mode = streamlit.checkbox("Draw (or Delete)?", True)

canvas_result = st_canvas(
    fill_color='#000000',
    stroke_width=20,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=SIZE,
    height=SIZE,
    drawing_mode="freedraw" if mode else "transform",
    key='canvas')

if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    streamlit.write('Model Input')
    streamlit.image(rescaled)

if streamlit.button('Predict'):
    test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # val = model.predict(test_x.reshape(1, 28, 28))
    # streamlit.write(f'result: {np.argmax(val[0])}')
    # streamlit.bar_chart(val[0])

# def preprocess_image(image):
#     transform = transforms.Compose([
#         transforms.Resize((28, 28)),
#         transforms.ToTensor(),
#     ])
#     image_resized = transform(image)
#     flat = torch.flatten(image_resized, start_dim=1)
#     return flat
#
# def predict(test_data):
#     model_uri = "models:/mnist_mlp/latest"
#     model = mlflow.pytorch.load_model(model_uri)
#     model = model.to(device)
#     model.eval()
#
#     inputs = preprocess_image(test_data["image"])
#
#     with torch.no_grad():
#         output = model(inputs)
#
#     return output

