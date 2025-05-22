import mlflow
import os
import tempfile
import zipfile
import shutil
import pickle
import requests
from typing import Any, Dict
from contextlib import contextmanager
import logging
import mimetypes
logger = logging.getLogger("FastAPITracker")
logger.setLevel(logging.INFO)  # Ou INFO si tu veux moins de verbosité

if not logger.handlers:  # Pour éviter d'ajouter plusieurs handlers
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class FastAPITracker:
    def __init__(self, tracking_uri: str):
        self.tracking_uri = tracking_uri.rstrip("/")
        self.run_id = None
        self.experiment_name = None
        logger.info(f"🔗 [Init] Tracking URI définie sur : {self.tracking_uri}")

    def set_experiment(self, experiment_name: str):
        self.experiment_name = experiment_name
        logger.info(f"🧪 [Expérience] Nom de l'expérience défini : {self.experiment_name}")

    @contextmanager
    def start_run(self, run_name: str = None):
        payload = {"experiment_name": self.experiment_name}
        if run_name:
            payload["run_name"] = run_name
        logger.info(f"🚀 [Run] Démarrage d’un run : {run_name or '(sans nom)'}")
        response = requests.post(f"{self.tracking_uri}/create-run", json=payload)
        response.raise_for_status()
        self.run_id = response.json()["run_id"]
        logger.info(f"✅ [Run] Run lancé avec l’ID : {self.run_id}")
        try:
            yield self
        finally:
            logger.info(f"🛑 [Run] Fin du run avec l’ID : {self.run_id}")
            self.run_id = None

    def log_params(self, params: Dict[str, Any]):
        logger.info(f"🛠️ [Paramètres] Enregistrement de {len(params)} paramètres : {params}")
        payload = {"run_id": self.run_id, "params": params}
        response = requests.post(f"{self.tracking_uri}/log-params", json=payload)
        response.raise_for_status()
        logger.info(f"✅ [Paramètres] Paramètres enregistrés avec succès.")

    def log_metrics(self, metrics: Dict[str, float]):
        logger.info(f"📊 [Métriques] Enregistrement de {len(metrics)} métriques : {metrics}")
        payload = {"run_id": self.run_id, "metrics": metrics}
        response = requests.post(f"{self.tracking_uri}/log-metrics", json=payload)
        response.raise_for_status()
        logger.info(f"✅ [Métriques] Métriques enregistrées avec succès.")

    def log_artifact(self, file_path: str, artifact_path: str = ""):
      logger.info(f"📊 [Artefact] Sauvegarde de {file_path} dans le dossier {artifact_path}")
      mimetype = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
      with open(file_path, "rb") as f:
          files = {
              "artifact_file": (os.path.basename(file_path), f, mimetype)
          }
          data = {
              "run_id": self.run_id,
              "artifact_path": artifact_path  # ce sera un dossier
          }
          response = requests.post(f"{self.tracking_uri}/log-artifact", data=data, files=files)
          response.raise_for_status()


    def log_model(self, model, artifact_path: str, model_type: str = "sklearn", input_example=None):
        logger.info(f"🧠 [Modèle] Sauvegarde du modèle ({model_type}) en cours...")
        if mlflow.active_run() is not None:
            mlflow.end_run()

        with tempfile.TemporaryDirectory() as tmpdir:
            with mlflow.start_run() as local_run:
                model_dir = os.path.join(tmpdir, "model_artifact")
                logger.info(f"📁 [Temp] Dossier temporaire créé : {model_dir}")

                if model_type == "sklearn":
                    mlflow.sklearn.save_model(sk_model=model, path=model_dir, input_example=input_example)
                elif model_type == "keras":
                    mlflow.keras.save_model(model, path=model_dir, input_example=input_example)
                elif model_type == "pyfunc":
                    mlflow.pyfunc.save_model(path=model_dir, python_model=model, input_example=input_example)
                else:
                    raise ValueError(f"🚫 [Erreur] Type de modèle non supporté : {model_type}")

                logger.info(f"✅ [Modèle] Modèle sauvegardé localement.")

            zip_path = os.path.join(tmpdir, "model.zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(model_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, model_dir)
                        zipf.write(file_path, arcname)

            logger.info(f"🗜️ [ZIP] Modèle compressé dans : {zip_path}")
            logger.debug(f"🔍 [DEBUG] Existe ? {os.path.exists(zip_path)}")
            logger.debug(f"📦 [DEBUG] Taille ZIP : {os.path.getsize(zip_path)} octets")
            logger.debug(f"✅ [DEBUG] ZIP valide ? {zipfile.is_zipfile(zip_path)}")

            with open(zip_path, "rb") as f:
                files = {"zipped_model": (f"{artifact_path}.zip", f, "application/zip")}
                data = {
                    "run_id": self.run_id,
                    "artifact_path": artifact_path,
                    "model_type": model_type
                }
                logger.info(f"📤 [Upload] Envoi du modèle à l’API pour enregistrement...")
                response = requests.post(f"{self.tracking_uri}/log-model", data=data, files=files)
                response.raise_for_status()
                logger.info(f"🎉 [Succès] Modèle envoyé et enregistré avec succès.")
