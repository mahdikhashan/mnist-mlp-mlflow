app:
	streamlit run app.py

local-registry:
	mlflow server --host 127.0.0.1 --port 8090

serve:
	flask --app serve run --port 5003
