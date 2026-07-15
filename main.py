"""
AI Flight Operations Copilot
Production Grade Local AI Aviation Assistant

Architecture:
- Local Ollama LLM
- ChromaDB RAG
- Sentence Transformer Embeddings
- Weather ML Prediction
- Agentic Routing System

Author: Aviation AI Engineering System
"""


# ============================================================
# SECTION 1
# CONFIGURATION, LOGGING, BOOTSTRAP & EXCEPTIONS
# ============================================================


import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import re
import csv
import json
import pickle
import random
import logging
import warnings

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


import numpy as np
import pandas as pd


from pydantic import BaseModel
from rich.console import Console
from rich.logging import RichHandler


warnings.filterwarnings("ignore")


# ------------------------------------------------------------
# Rich Console
# ------------------------------------------------------------

console = Console()


# ------------------------------------------------------------
# Custom Exceptions
# ------------------------------------------------------------


class AviationCopilotException(Exception):
    """Base aviation copilot exception"""
    pass



class LLMConnectionError(AviationCopilotException):
    pass



class WeatherParserError(AviationCopilotException):
    pass



class RAGQueryError(AviationCopilotException):
    pass



class DataBootstrapError(AviationCopilotException):
    pass



# ------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------


LOG_FILE = "flight_ops_copilot.log"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        RichHandler(console=console),
        logging.FileHandler(LOG_FILE)
    ]
)


logger = logging.getLogger("FlightOpsCopilot")



# ------------------------------------------------------------
# Application Settings
# ------------------------------------------------------------


class CopilotSettings(BaseModel):

    data_directory: str = "./data"

    pdf_directory: str = "./data"

    weather_dataset: str = "./data/weatherHistory.csv"

    chroma_directory: str = "./data/chroma_db"

    weather_model_file: str = "./data/weather_model.pkl"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    ollama_model: str = "qwen3:8b"


settings = CopilotSettings()



# ============================================================
# DATA BOOTSTRAPPING
# ============================================================


def create_flight_manual():
    # Deprecated: Using PDFs from data directory instead
    pass



def generate_weather_dataset():

    regions = [
        "Northeast",
        "MiddleEast",
        "Europe",
        "Asia",
        "NorthAmerica"
    ]


    records = []


    for _ in range(1000):

        region = random.choice(regions)


        wind = random.randint(0,40)

        visibility = random.randint(1,10)

        temperature = random.randint(-10,45)

        ceiling = random.randint(500,20000)


        if visibility < 3 or ceiling < 1500 or wind > 30:

            delay = "High"


        elif visibility < 6 or ceiling < 5000:

            delay = "Medium"


        else:

            delay = "Low"



        metar = (
            f"XXXX "
            f"{random.randint(1000,2300)}Z "
            f"{wind:03d}KT "
            f"{visibility}SM "
            f"SCT{ceiling//100} "
            f"{temperature}/12"
        )


        records.append(
            {
                "metar": metar,
                "region": region,
                "wind_speed": wind,
                "visibility": visibility,
                "temperature": temperature,
                "ceiling": ceiling,
                "delay_risk": delay
            }
        )



    df = pd.DataFrame(records)


    df.to_csv(
        settings.weather_dataset,
        index=False
    )



def bootstrap_data_environment():

    try:

        data_path = Path(settings.data_directory)


        if not data_path.exists():

            data_path.mkdir(
                parents=True
            )


        chroma_path = Path(
            settings.chroma_directory
        )


        chroma_path.mkdir(
            exist_ok=True
        )


        if not Path(settings.weather_dataset).exists():

            logger.info(
                "Generating weather ML dataset..."
            )

            generate_weather_dataset()


        airports_file = data_path / "airports.csv"
        if not airports_file.exists():
            import urllib.request
            logger.info("Downloading airports.csv from OurAirports...")
            url = "https://davidmegginson.github.io/ourairports-data/airports.csv"
            urllib.request.urlretrieve(url, airports_file)
            logger.info("Airport database download complete.")


        logger.info(
            "Data environment ready"
        )



    except Exception as exc:

        raise DataBootstrapError(
            str(exc)
        )



# ============================================================
# SECTION 2
# AVIATION WEATHER NLP PARSER FOUNDATION
# ============================================================


class METARFeatures(BaseModel):

    wind_speed: float

    wind_gust: Optional[float] = None

    visibility: float

    temperature: float

    dewpoint: Optional[float] = None

    ceiling: float

    severe_icing_alert: bool = False

    severe_turbulence_alert: bool = False

    report_type: str = "METAR"

    region: str



class AviationWeatherParser:


    def parse(
        self,
        metar: str,
        region: str="Unknown"
    ) -> METARFeatures:


        try:
            # 1. SIGMET Report Detection
            if any(x in metar.upper() for x in ["SIGMET", "WSUS", "WCCL", "WSTU", "WV2", "WV3"]):
                severe_icing_alert = "SEV ICE" in metar.upper() or "FZRA" in metar.upper()
                severe_turbulence_alert = "SEV TURB" in metar.upper() or "TURB" in metar.upper() or "CAT" in metar.upper()
                has_volcanic_ash = "VA" in metar.upper() or "ASH" in metar.upper()
                has_severe_ts = "TS" in metar.upper() or "SQL" in metar.upper()

                # Model severe hazard environment to trigger high-risk ML predictions
                wind = 40.0 if (severe_turbulence_alert or has_severe_ts) else 10.0
                wind_gust = 50.0 if wind == 40.0 else None
                visibility = 0.5 if (severe_icing_alert or has_volcanic_ash) else 10.0
                ceiling = 200.0 if (severe_icing_alert or severe_turbulence_alert or has_volcanic_ash) else 99999.0
                temperature = -5.0 if severe_icing_alert else 15.0

                return METARFeatures(
                    wind_speed=wind,
                    wind_gust=wind_gust,
                    visibility=visibility,
                    temperature=temperature,
                    dewpoint=None,
                    ceiling=ceiling,
                    severe_icing_alert=severe_icing_alert,
                    severe_turbulence_alert=severe_turbulence_alert,
                    report_type="SIGMET",
                    region=region
                )

            # 2. TAF Forecast Report Detection
            if "TAF" in metar.upper():
                # Extract worst-case wind across forecast periods
                winds = [0.0]
                gusts = [0.0]
                for wm in re.finditer(r"\b(?:\d{3}|VRB)(\d{2})(?:G(\d{2}))?(?:KT|MPS)\b", metar, re.I):
                    speed = float(wm.group(1))
                    gust = float(wm.group(2)) if wm.group(2) else 0.0
                    if "MPS" in wm.group(0).upper():
                        speed = speed * 1.94384
                        gust = gust * 1.94384
                    winds.append(speed)
                    gusts.append(gust)
                wind = max(winds)
                wind_gust = max(gusts) if max(gusts) > 0.0 else None

                # Extract worst-case visibility (minimum forecast visibility)
                visibilities = [10.0]
                for vm in re.finditer(r"\b(\d+(?:\s+\d+/\d+)?|\d+/\d+|\d+)SM\b", metar, re.I):
                    vis_str = vm.group(1)
                    if "/" in vis_str:
                        if " " in vis_str:
                            parts = vis_str.split()
                            whole = float(parts[0])
                            frac_parts = parts[1].split("/")
                            visibilities.append(whole + float(frac_parts[0])/float(frac_parts[1]))
                        else:
                            frac_parts = vis_str.split("/")
                            visibilities.append(float(frac_parts[0])/float(frac_parts[1]))
                    else:
                        visibilities.append(float(vis_str))
                for mm in re.finditer(r"(?:KT|MPS)\s+(\d{4})\b", metar, re.I):
                    visibilities.append(float(mm.group(1)) * 0.000621371)
                visibility = min(visibilities)

                # Extract worst-case ceiling (lowest forecast ceiling)
                ceilings = [99999.0]
                for cm in re.finditer(r"\b(?:BKN|OVC|VV)(\d{3})", metar, re.I):
                    ceilings.append(float(cm.group(1)) * 100)
                ceiling = min(ceilings)

                # Extract worst-case forecast minimum temperature
                temperature = 20.0
                dewpoint = None
                tn_match = re.search(r"\bTN(M?\d{2})/", metar, re.I)
                if tn_match:
                    tn_str = tn_match.group(1)
                    if tn_str.upper().startswith("M"):
                        temperature = -float(tn_str[1:])
                    else:
                        temperature = float(tn_str)

                severe_icing_alert = False
                if temperature < 0.0 or any(code in metar.upper() for code in ["-SN", "BLSN", "FZFG", "FZRA", "FZDZ"]):
                    has_freezing = any(code in metar.upper() for code in ["FZ", "SN", "BLSN", "PL"])
                    if has_freezing:
                        severe_icing_alert = True
                        if temperature > 0.0:
                            temperature = -2.0  # Force sub-zero to ensure ML icing alignment

                return METARFeatures(
                    wind_speed=wind,
                    wind_gust=wind_gust,
                    visibility=visibility,
                    temperature=temperature,
                    dewpoint=dewpoint,
                    ceiling=ceiling,
                    severe_icing_alert=severe_icing_alert,
                    severe_turbulence_alert=False,
                    report_type="TAF",
                    region=region
                )

            # 3. Standard METAR Parsing (Default)
            # Wind: match 3 digits or VRB, then 2 digits speed, optional gust (G followed by 2 digits), and KT or MPS
            wind_match = re.search(
                r"\b(?:\d{3}|VRB)(\d{2})(?:G(\d{2}))?(?:KT|MPS)\b",
                metar,
                re.I
            )

            wind = 0
            wind_gust = None
            if wind_match:
                wind_speed_raw = float(wind_match.group(1))
                wind_gust_raw = float(wind_match.group(2)) if wind_match.group(2) else None
                
                if "MPS" in wind_match.group(0).upper():
                    wind = round(wind_speed_raw * 1.94384, 2)
                    if wind_gust_raw is not None:
                        wind_gust = round(wind_gust_raw * 1.94384, 2)
                else:
                    wind = wind_speed_raw
                    wind_gust = wind_gust_raw

            # CAVOK check
            if "CAVOK" in metar.upper():
                visibility = 10.0
                ceiling = 99999.0
            else:
                # Visibility: match US format (e.g. 2 1/2SM, 10SM) or metric format (4-digit number e.g. 4000, 9999)
                vis_match = re.search(
                    r"\b(\d+(?:\s+\d+/\d+)?|\d+/\d+|\d+)SM\b",
                    metar,
                    re.I
                )
                if vis_match:
                    vis_str = vis_match.group(1)
                    if "/" in vis_str:
                        if " " in vis_str:
                            parts = vis_str.split()
                            whole = float(parts[0])
                            frac_parts = parts[1].split("/")
                            frac = float(frac_parts[0]) / float(frac_parts[1])
                            visibility = whole + frac
                        else:
                            frac_parts = vis_str.split("/")
                            visibility = float(frac_parts[0]) / float(frac_parts[1])
                    else:
                        visibility = float(vis_str)
                else:
                    # Look for 4-digit metric visibility group (e.g., 4000, 9999) following wind speed group
                    metric_match = re.search(
                        r"(?:KT|MPS)\s+(\d{4})\b",
                        metar,
                        re.I
                    )
                    if metric_match:
                        # Convert meters to Statute Miles (1 meter = 0.000621371 SM)
                        visibility = round(float(metric_match.group(1)) * 0.000621371, 2)
                    else:
                        visibility = 10.0

                # Ceiling: lowest broken (BKN) or overcast (OVC) layer, or vertical visibility (VV)
                # Removed trailing \b to allow suffixes like CB or TCU directly attached
                ceiling_match = re.search(
                    r"\b(?:BKN|OVC|VV)(\d{3})",
                    metar,
                    re.I
                )
                ceiling = 99999.0
                if ceiling_match:
                    ceiling = float(ceiling_match.group(1)) * 100

            # Temperature and Dewpoint: METAR uses 'M' for negative numbers (e.g. M02/M05)
            # We match 2 digits to avoid matching fractional visibility like 1/2SM
            temp_match = re.search(
                r"\b(M?\d{2})/(M?\d{2})?\b",
                metar,
                re.I
            )

            temperature = 20.0
            dewpoint = None
            if temp_match:
                temp_str = temp_match.group(1)
                if temp_str.upper().startswith("M"):
                    temperature = -float(temp_str[1:])
                else:
                    temperature = float(temp_str)
                
                dp_str = temp_match.group(2)
                if dp_str:
                    if dp_str.upper().startswith("M"):
                        dewpoint = -float(dp_str[1:])
                    else:
                        dewpoint = float(dp_str)

            # Severe Icing Alert check: trigger whenever FZ, -SN, or BLSN appear alongside sub-zero temperatures
            severe_icing_alert = False
            if temperature < 0.0:
                has_freezing_phenomena = any(code in metar.upper() for code in ["FZ", "-SN", "BLSN", "SN", "PL"])
                if has_freezing_phenomena:
                    severe_icing_alert = True

            return METARFeatures(
                wind_speed=wind,
                wind_gust=wind_gust,
                visibility=visibility,
                temperature=temperature,
                dewpoint=dewpoint,
                ceiling=ceiling,
                severe_icing_alert=severe_icing_alert,
                severe_turbulence_alert=False,
                report_type="METAR",
                region=region
            )


        except Exception as exc:


            raise WeatherParserError(
                f"METAR parsing failed: {exc}"
            )



# ============================================================
# TEST BOOTSTRAP
# ============================================================


bootstrap_data_environment()


logger.info(
    "PART 1 INITIALIZATION COMPLETE"
)

# ============================================================
# SECTION 3
# AVIATION WEATHER ML ENGINE
# ============================================================


from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

import joblib



class AviationWeatherML:


    def __init__(self):

        self.parser = AviationWeatherParser()

        self.model_path = (
            settings.weather_model_file
        )

        self.pipeline = None



    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    def train(self):

        try:

            logger.info(
                "Training weather prediction model..."
            )


            df = pd.read_csv(
                settings.weather_dataset
            )

            # Extract features and convert metric to standard aviation units (Knots, SM)
            X = pd.DataFrame()
            X['temperature'] = pd.to_numeric(
                df['Temperature (C)'],
                errors='coerce'
            ).fillna(10)
            
            wind_kmh = pd.to_numeric(
                df['Wind Speed (km/h)'],
                errors='coerce'
            ).fillna(0)
            X['wind_speed'] = wind_kmh / 1.852
            
            vis_km = pd.to_numeric(
                df['Visibility (km)'],
                errors='coerce'
            ).fillna(10)
            X['visibility'] = vis_km / 1.60934

            # Create synthetic delay_risk target based on FAA flight rules and weather conditions
            y = []
            for idx, row in X.iterrows():
                humidity = df.loc[idx, 'Humidity'] if 'Humidity' in df.columns else 0.8
                # Approximate dewpoint spread: spread = (1 - RH) * 20
                spread = (1.0 - humidity) * 20.0
                
                # FAA Rules:
                # LIFR: visibility < 1 SM
                # IFR: visibility < 3 SM
                # MVFR: visibility < 5 SM
                # VFR: visibility >= 5 SM
                
                is_lifr = row['visibility'] < 1.0
                is_ifr = row['visibility'] < 3.0
                is_mvfr = row['visibility'] < 5.0
                
                is_severe_wind = row['wind_speed'] > 30.0
                is_extreme_temp = row['temperature'] < -20.0 or row['temperature'] > 43.0
                is_icing_risk = row['temperature'] < 3.0 and spread <= 3.0
                
                # Check for active severe weather summaries in the dataset
                summary = str(df.loc[idx, 'Summary']).lower()
                is_severe_summary = any(s in summary for s in ["heavy rain", "thunderstorm", "storm", "sleet", "ice", "snow", "windy"])
                
                if is_lifr or is_severe_wind or is_extreme_temp or is_icing_risk or is_severe_summary:
                    y.append("High")
                elif is_ifr or is_mvfr or row['wind_speed'] > 18.0 or spread <= 3.0:
                    y.append("Medium")
                else:
                    y.append("Low")

            y = pd.Series(y)

            # Fit train/test split to report real validation accuracy
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            classifier = RandomForestClassifier(

                n_estimators=100,

                random_state=42

            )

            # Put StandardScaler inside the pipeline steps
            self.pipeline = Pipeline(

                steps=[
                    (
                        "scaler",
                        StandardScaler()
                    ),
                    (
                        "classifier",
                        classifier
                    )

                ]

            )


            self.pipeline.fit(
                X_train,
                y_train
            )



            predictions = self.pipeline.predict(X_val)


            accuracy = accuracy_score(
                y_val,
                predictions
            )



            logger.info(
                f"Weather ML Validation Accuracy: {accuracy:.4f}"
            )

            # Re-fit pipeline on all data for production
            self.pipeline.fit(X, y)

            joblib.dump(

                self.pipeline,

                self.model_path

            )


        except Exception as exc:

            logger.exception(exc)

            raise AviationCopilotException(
                "Weather model training failed"
            )



    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    def load_or_train(self):

        if Path(
            self.model_path
        ).exists():


            logger.info(
                "Loading weather ML model"
            )

            try:
                self.pipeline = joblib.load(
                    self.model_path
                )
            except (EOFError, Exception) as e:
                logger.warning(
                    f"Failed to load model: {e}. Training new model..."
                )
                self.train()


        else:

            self.train()



    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    def predict(
        self,
        metar:str,
        region:str
    ):


        try:


            if self.pipeline is None:

                self.load_or_train()



            features = self.parser.parse(

                metar,

                region

            )


            input_df = pd.DataFrame(
                [
                    {

                    "temperature":
                    features.temperature,

                    "wind_speed":
                    features.wind_speed,


                    "visibility":
                    features.visibility

                    }
                ]
            )



            result = self.pipeline.predict(
                input_df
            )[0]



            probability = max(
                self.pipeline.predict_proba(
                    input_df
                )[0]
            )



            spread = None
            if features.temperature is not None and features.dewpoint is not None:
                spread = round(features.temperature - features.dewpoint, 2)

            return {

                "risk":
                result,


                "confidence":
                round(
                    probability,
                    3
                ),


                "features":
                features.model_dump(),

                "temperature_dewpoint_spread": spread

            }


        except Exception as exc:


            raise AviationCopilotException(
                str(exc)
            )





# ============================================================
# SECTION 4
# LOCAL CHROMADB RAG ENGINE
# ============================================================


import chromadb


from sentence_transformers import SentenceTransformer




class LocalEmbedding:


    def __init__(self):


        self.model = SentenceTransformer(

            settings.embedding_model

        )



    def encode(
        self,
        texts
    ):

        return self.model.encode(

            texts,

            show_progress_bar=False

        ).tolist()





class LocalRAGEngine:


    def __init__(self):


        self.client = chromadb.PersistentClient(

            path=settings.chroma_directory

        )


        self.embedding = LocalEmbedding()



        self.collection = (

            self.client.get_or_create_collection(

                name="aviation_manuals_v6"

            )

        )

        self.airports_collection = (

            self.client.get_or_create_collection(

                name="aviation_airports_v6"

            )

        )



    # --------------------------------------------------------
    # Split documents
    # --------------------------------------------------------

    def chunk_document(
        self,
        text,
        size=500
    ):


        chunks=[]


        words=text.split()


        for i in range(
            0,
            len(words),
            size
        ):

            chunks.append(

                " ".join(
                    words[i:i+size]
                )

            )


        return chunks



    # --------------------------------------------------------
    # Ingest manuals
    # --------------------------------------------------------

    def ingest(self):


        try:


            existing_manuals = (

                self.collection.count()

            )

            existing_airports = (

                self.airports_collection.count()

            )


            if existing_manuals >= 3400 and existing_airports >= 80000:
                logger.info("RAG database already populated with metadata-enriched chunks.")
                return
            else:
                logger.info("RAG database incomplete or missing. Resetting collections for full ingestion...")
                try:
                    self.client.delete_collection(name="aviation_manuals_v6")
                except Exception:
                    pass
                try:
                    self.client.delete_collection(name="aviation_airports_v6")
                except Exception:
                    pass
                self.collection = self.client.create_collection(name="aviation_manuals_v6")
                self.airports_collection = self.client.create_collection(name="aviation_airports_v6")



            logger.info(
                "Creating aviation knowledge base from PDFs and glossaries..."
            )

            from pypdf import PdfReader
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            documents = []
            metadatas = []
            ids = []
            chunk_id = 0

            pdf_directory = Path(settings.pdf_directory)

            # 1. Load PDFs page-by-page
            for pdf_file in pdf_directory.glob("*.pdf"):
                try:
                    logger.info(f"Loading PDF: {pdf_file.name}")
                    reader = PdfReader(pdf_file)
                    for page_idx, page in enumerate(reader.pages):
                        page_text = page.extract_text()
                        if not page_text or not page_text.strip():
                            continue
                        page_chunks = text_splitter.split_text(page_text)
                        for chunk in page_chunks:
                            documents.append(chunk)
                            metadatas.append({
                                "source": pdf_file.name,
                                "page": page_idx + 1
                            })
                            ids.append(f"doc_{pdf_file.stem}_{chunk_id}")
                            chunk_id += 1
                except Exception as e:
                    logger.warning(f"Failed to load {pdf_file.name}: {e}")
                    continue

            # 2. Load Markdown files (like weather_glossary.md)
            for md_file in pdf_directory.glob("*.md"):
                try:
                    logger.info(f"Loading Markdown file: {md_file.name}")
                    with open(md_file, "r", encoding="utf-8") as f:
                        md_text = f.read()
                    if not md_text.strip():
                        continue
                    md_chunks = text_splitter.split_text(md_text)
                    for chunk in md_chunks:
                        documents.append(chunk)
                        metadatas.append({
                            "source": md_file.name,
                            "page": 1
                        })
                        ids.append(f"doc_{md_file.stem}_{chunk_id}")
                        chunk_id += 1
                except Exception as e:
                    logger.warning(f"Failed to load {md_file.name}: {e}")
                    continue

            # Index documents in batches
            if documents:
                embeddings = self.embedding.encode(documents)
                batch_size = 500
                for i in range(0, len(documents), batch_size):
                    end_i = min(i + batch_size, len(documents))
                    self.collection.add(
                        documents=documents[i:end_i],
                        embeddings=embeddings[i:end_i],
                        metadatas=metadatas[i:end_i],
                        ids=ids[i:end_i]
                    )
                logger.info(f"Indexed {len(documents)} document chunks from manuals and glossaries.")

            # 3. Load and index airports from airports.csv
            airports_file = Path(settings.data_directory) / "airports.csv"
            runways_file = Path(settings.data_directory) / "runways.csv"
            
            if not runways_file.exists():
                logger.info("Downloading runways.csv from OurAirports...")
                import urllib.request
                url = "https://davidmegginson.github.io/ourairports-data/runways.csv"
                try:
                    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req, timeout=30) as response, open(runways_file, 'wb') as out_file:
                        out_file.write(response.read())
                    logger.info("runways.csv successfully downloaded.")
                except Exception as e:
                    logger.warning(f"Failed to download runways.csv: {e}")

            runway_dict = {}
            if runways_file.exists():
                try:
                    logger.info("Parsing runways.csv...")
                    runways_df = pd.read_csv(runways_file)
                    runways_df = runways_df[runways_df['airport_ident'].notna()]
                    for _, row in runways_df.iterrows():
                        ident = str(row['airport_ident']).upper()
                        le_ident = str(row['le_ident']) if pd.notna(row['le_ident']) else ""
                        he_ident = str(row['he_ident']) if pd.notna(row['he_ident']) else ""
                        le_heading = f"{int(float(row['le_heading_degT']))}°" if pd.notna(row['le_heading_degT']) else ""
                        he_heading = f"{int(float(row['he_heading_degT']))}°" if pd.notna(row['he_heading_degT']) else ""
                        
                        rw_str = f"Runway {le_ident}/{he_ident}"
                        details = []
                        if le_heading:
                            details.append(f"{le_ident}: {le_heading}")
                        if he_heading:
                            details.append(f"{he_ident}: {he_heading}")
                        if details:
                            rw_str += f" (headings: {', '.join(details)})"
                        
                        if ident not in runway_dict:
                            runway_dict[ident] = []
                        runway_dict[ident].append(rw_str)
                    logger.info(f"Loaded runway details for {len(runway_dict)} airports.")
                except Exception as e:
                    logger.warning(f"Failed to parse runways.csv: {e}")

            if airports_file.exists():
                logger.info("Loading and indexing all airport codes from airports.csv...")
                airports_df = pd.read_csv(airports_file)
                # Ingest all airports that have a valid identifier
                filtered_airports = airports_df[airports_df['ident'].notna()]
                
                airport_docs = []
                airport_metadatas = []
                airport_ids = []
                
                for _, row in filtered_airports.iterrows():
                    ident = str(row['ident']).upper()
                    name = str(row['name'])
                    municipality = str(row['municipality']) if pd.notna(row['municipality']) else "Unknown"
                    country = str(row['iso_country']) if pd.notna(row['iso_country']) else "Unknown"
                    elevation = f"{row['elevation_ft']} ft" if pd.notna(row['elevation_ft']) else "Unknown"
                    iata = str(row['iata_code']) if pd.notna(row['iata_code']) else "N/A"
                    icao = str(row['gps_code']) if pd.notna(row['gps_code']) else ident
                    
                    runways_info = "N/A"
                    if ident in runway_dict:
                        runways_info = ", ".join(runway_dict[ident])
                    
                    doc_text = (
                        f"Airport Code Information:\n"
                        f"- Identifier (ICAO): {ident}\n"
                        f"- Name: {name}\n"
                        f"- Municipality/City: {municipality}\n"
                        f"- Country: {country}\n"
                        f"- Elevation: {elevation}\n"
                        f"- IATA Code: {iata}\n"
                        f"- GPS Code: {icao}\n"
                        f"- Type: {row['type']}\n"
                        f"- Runways: {runways_info}"
                    )
                    airport_docs.append(doc_text)
                    airport_metadatas.append({
                        "source": "airports.csv",
                        "page": 1,
                        "ident": ident,
                        "iata": iata,
                        "country": country
                    })
                    airport_ids.append(f"airport_{ident}")
                
                # Index airports in batches
                batch_size = 2000
                logger.info(f"Indexing {len(airport_docs)} airports into ChromaDB in batches with optimized dummy embeddings...")
                for idx in range(0, len(airport_docs), batch_size):
                    end_idx = min(idx + batch_size, len(airport_docs))
                    batch_docs = airport_docs[idx:end_idx]
                    batch_meta = airport_metadatas[idx:end_idx]
                    batch_ids = airport_ids[idx:end_idx]
                    batch_embs = [[0.0] * 384 for _ in range(len(batch_docs))]
                    self.airports_collection.add(
                        documents=batch_docs,
                        embeddings=batch_embs,
                        metadatas=batch_meta,
                        ids=batch_ids
                    )
                logger.info(f"Indexed all {len(airport_docs)} airports from airports.csv.")



        except Exception as exc:


            logger.exception(exc)


            raise RAGQueryError(
                "Document ingestion failed"
            )




    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    def retrieve(
        self,
        query,
        top_k=3
    ):


        try:


            # Extract all 3-4 letter uppercase words as potential airport codes
            possible_codes = re.findall(r"\b([A-Z]{3,4})\b", query.upper())
            
            # Common query words that should not be treated as airport codes
            EXCLUDED_WORDS = {
                "WHAT", "HOW", "THE", "FOR", "AND", "YOU", "ARE", "GET", "RUN", 
                "WIND", "TEMP", "DATE", "TIME", "ZONE", "CODE", "INFO", "THIS", 
                "THAT", "WITH", "FROM", "YOUR", "WILL", "HAVE", "SOME", "MORE", 
                "ABOUT", "LIKE", "HERE", "WANT", "NEED", "FUEL", "BURN", "LAND", 
                "STOP", "RISK", "EVAL", "TEST", "PLAN", "SHOW", "LIST", "VIEW", 
                "CHECK", "RULE", "DATA", "FILE", "PATH", "TYPE", "NAME", "CITY",
                "TAF", "TEMPO", "BECMG", "FM", "PROB", "PROB30", "PROB40", "SIGMET",
                "METAR", "FZFG", "BLSN", "FZRA", "OVC", "BKN", "SCT", "FEW", "CAVOK"
            }
            
            filtered_codes = [c for c in possible_codes if c not in EXCLUDED_WORDS]
            results = None
            
            for code in filtered_codes:
                if len(code) == 4:
                    results = self.airports_collection.get(
                        where={"ident": code}
                    )
                elif len(code) == 3:
                    results = self.airports_collection.get(
                        where={"iata": code}
                    )
                
                # If we found exact matches using metadata filters, use them!
                if results and results.get("documents") and results["documents"]:
                    break
            
            # If we found exact matches using metadata filters, use them!
            if results and results.get("documents") and results["documents"]:
                documents = results["documents"]
                metadatas = results["metadatas"]
                retrieved_chunks = []
                for doc, meta in zip(documents, metadatas):
                    source = meta.get("source", "Unknown")
                    page = meta.get("page", 1)
                    retrieved_chunks.append(f"[Source: {source}, Page: {page}]\n{doc}")
                return "\n\n".join(retrieved_chunks)

            # Fallback to standard vector search
            query_embedding = (

                self.embedding.encode(
                    [query]
                )[0]

            )



            result = self.collection.query(

                query_embeddings=[
                    query_embedding
                ],

                n_results=top_k

            )



            documents = result.get("documents", [])
            metadatas = result.get("metadatas", [])



            if documents and documents[0]:

                retrieved_chunks = []
                for doc, meta in zip(documents[0], metadatas[0]):
                    source = meta.get("source", "Unknown")
                    page = meta.get("page", 1)
                    retrieved_chunks.append(f"[Source: {source}, Page: {page}]\n{doc}")
                return "\n\n".join(retrieved_chunks)


            return "No relevant aviation manual or airport information found."



        except Exception as exc:


            raise RAGQueryError(
                str(exc)
            )





# ============================================================
# INITIALIZE PART 2 COMPONENTS
# ============================================================


weather_engine = AviationWeatherML()


rag_engine = LocalRAGEngine()



weather_engine.load_or_train()


rag_engine.ingest()



logger.info(
    "PART 2 INITIALIZATION COMPLETE"
)

# ============================================================
# SECTION 5
# LOCAL LLM INTERFACE
# ============================================================


import ollama


class LocalLLM:


    def __init__(self):

        self.model = settings.ollama_model
        # Groq client integration for cloud-based demo deployment
        self.groq_api_key = os.environ.get("GROQ_API_KEY")
        self.groq_client = None
        if self.groq_api_key:
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=self.groq_api_key)
                self.groq_model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
                logger.info(f"Groq API client initialized using model: {self.groq_model}")
            except Exception as e:
                logger.warning(f"Failed to load Groq client: {e}")



    # --------------------------------------------------------
    # Ollama generation
    # --------------------------------------------------------

    def generate(
        self,
        prompt:str
    ):

        if self.groq_client:
            try:
                response = self.groq_client.chat.completions.create(
                    model=self.groq_model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
                return response.choices[0].message.content
            except Exception as exc:
                logger.warning(f"Groq API call failed: {exc}. Falling back to Ollama...")

        try:

            response = ollama.chat(

                model=self.model,

                messages=[

                    {
                        "role":"user",
                        "content":prompt
                    }

                ]

            )


            return response[
                "message"
            ][
                "content"
            ]



        except Exception as exc:


            logger.warning(
                f"Ollama unavailable: {exc}"
            )


            return self.huggingface_fallback(
                prompt
            )




    # --------------------------------------------------------
    # Lightweight offline fallback
    # --------------------------------------------------------

    def huggingface_fallback(
        self,
        prompt
    ):


        try:


            from transformers import pipeline



            generator = pipeline(

                "text-generation",

                model="distilgpt2",

                max_new_tokens=120

            )


            output = generator(
                prompt
            )


            return output[0][
                "generated_text"
            ]



        except Exception:


            return self.rule_based_response(
                prompt
            )




    # --------------------------------------------------------
    # Deterministic fallback
    # --------------------------------------------------------

    def rule_based_response(
        self,
        prompt
    ):


        return (

        "LOCAL AI FALLBACK MODE\n\n"

        "The language model is unavailable.\n"

        "Operational recommendation:\n"

        "- Verify aircraft status\n"

        "- Check METAR/TAF conditions\n"

        "- Review NOTAM restrictions\n"

        "- Consult approved aircraft manuals\n"

        )






# ============================================================
# SECTION 6
# FLIGHT OPERATIONS TOOLS
# ============================================================



class FlightCalculator:



    def fuel_burn(

        self,

        distance_nm:float,

        burn_rate:float=2600

    ):


        fuel = (

            distance_nm *
            burn_rate /
            60

        )


        return {

            "distance_nm":
            distance_nm,

            "estimated_fuel_kg":
            round(
                fuel,
                2
            )

        }



    def landing_distance(

        self,

        weight:float,

        wind:float

    ):


        base = 1500


        adjustment = (

            weight *
            0.8

        )


        wind_effect = (

            wind *
            15

        )


        return {

            "required_landing_distance_ft":

            round(

                base +
                adjustment -
                wind_effect,

                2

            )

        }

    def verify_weight_limits(
        self,
        zfw: float,
        fuel: float,
        payload: float,
        mtow: float = 79000.0
    ):
        tow = zfw + fuel + payload
        is_legal = tow <= mtow
        warning = ""
        if not is_legal:
            warning = "**CRITICAL WEIGHT VIOLATION - AIRCRAFT OVERWEIGHT FOR TAKEOFF**"
        return {
            "zero_fuel_weight_kg": zfw,
            "fuel_weight_kg": fuel,
            "payload_weight_kg": payload,
            "takeoff_weight_kg": tow,
            "max_takeoff_weight_kg": mtow,
            "is_legal_takeoff": is_legal,
            "weight_warning": warning
        }






# ============================================================
# SECTION 7
# AGENTIC ROUTING SYSTEM
# ============================================================



WEATHER_AGENT_SYSTEM_PROMPT = """ROLE: 
You are the Meteorological Analyst Specialist for an Airline-Grade Multi-Agent Flight Operations Copilot. Your job is to analyze raw aviation weather reports (METAR/TAF) and generate a safety-critical pilot briefing.

OPERATIONAL INSTRUCTIONS & CONSTRAINTS:
1. TIMELINE DECOUPLING: When evaluating TAF forecast periods (indicated by FM, TEMPO, or PROB30), treat each time block as its own independent weather environment. Do not apply the temperature of the initial daytime observation to a freezing forecast block hours later. Never write that the airport is "currently experiencing" a hazard if that hazard is only predicted in a future forecast block.
2. GEOGRAPHIC ISOLATION: Stick strictly to the environment of the requested airport ICAO code. Never pull, mention, or cross-contaminate runway data, layouts, or headings belonging to other global airports (e.g., if analyzing EHAM, do not reference runways or rules for KJFK or KORD).
3. AVIATION ABBREVIATION DICTIONARY: Standard aviation decoding vocabulary must be strictly enforced. Correctly decode 'BR' as Mist (not Brilliant Reflectivity), 'HZ' as Haze, and '-SN'/'SN' as Light Snow/Snow.
4. ICING HAZARDS: If you detect active freezing or frozen moisture codes (such as FZFG, FZRA, SN, -SN, or BLSN) alongside temperatures at or below 0°C, you must explicitly flag a high-priority "Severe Ground Icing Alert" and recommend mandatory Type I/IV fluid de-icing procedures.
5. WIND LIMITATIONS: Actively parse the wind speeds and gust thresholds (indicated by 'G'). If gusts exceed 20 knots, explicitly note the threat of severe wind gusts and advise cross-checking aircraft-specific crosswind limits.

OUTPUT FORMAT REQUIREMENTS:
- Use clear markdown headers (### Safety Considerations, ### Operational Recommendations, ### Limitations).
- Keep your sentences highly concise, professional, and tailored for flight crews under high-pressure scenarios."""

PERFORMANCE_AGENT_SYSTEM_PROMPT = """ROLE:
You are the Aircraft Performance and Weight & Balance Engineer Specialist for the Multi-Agent Flight Copilot pipeline. Your job is to calculate structural weights, environmental degradation factors, and legal runway stopping requirements.

CRITICAL PERFORMANCE & WEIGHT CONSTRAINTS:
1. TAXI VS. TAKEOFF WEIGHT LOGIC: Taxi-out fuel burn is a small, realistic margin (typically 200 kg to 500 kg for commercial narrow-body aircraft like the Boeing 737-800). Never assume or calculate that an aircraft burns large percentages or halves of its fuel tank during the taxi-out or takeoff phases.
2. TOW VS. MTOW GO/NO-GO GATE: If the calculated Takeoff Weight (TOW) exceeds the Maximum Structural Takeoff Weight (MTOW), you MUST issue a prominent, high-priority bold warning: "**CRITICAL WEIGHT VIOLATION - AIRCRAFT OVERWEIGHT FOR TAKEOFF**". Never minimize or justify a takeoff weight violation by pointing to a compliant landing weight or fuel burn. If TOW > MTOW, the flight is legally grounded.
3. EMPIRICAL FORMULA UNIT DISCIPLINE: When solving empirical performance formulas (such as runway landing distance equations), treat the variables (Weight, Wind, Temp) as pure, unitless scalars during the math execution steps. Do not append strings like "kg" or "KT" inside the middle of math steps, which leads to nonsense algebraic errors (e.g., adding kg to knots). The final computed value must strictly resolve to the single target measurement unit (e.g., Feet or Meters).
4. PERFORMANCE RETRIEVAL BOUNDS: Do not perform manual wind geometry or relative crosswind angle math here; rely entirely on the outputs generated by the crosswind mathematical helper modules or explicitly state the missing variables.

OUTPUT FORMAT REQUIREMENTS:
- Present all structural weight breakdowns (ZFW, Fuel, Payload, TOW, MTOW) in a clean, scannable Markdown Table for rapid verification."""

VALIDATOR_AGENT_SYSTEM_PROMPT = """ROLE:
You are the Automated Safety Inspector and Quality Control Guardrail Agent. You intercept the combined output draft generated by the specialized agents and audit it against raw telemetry data.

MANDATORY ANTI-CONFLATION & SAFETY RULES:
1. RUNWAY VISUAL RANGE (RVR) COGNIZANCE: RVR (e.g., R18R/0800FT) dictates horizontal visibility parameters down the runway centerline for instrument approach categories. It has ZERO correlation with tire hydroplaning, braking action coefficients, or water/slush depth. If you find any instance where the draft associates low RVR values with braking degradation or braking performance risks, you must strip or correct that text immediately.
2. ACCURATE CALCULATED CROSSWINDS: Cross-reference any crosswind assertions with deterministic geometry: Crosswind = Gust Speed * sin(Relative Angle). If a draft claims a wind closely aligned with a runway heading (e.g., 10-degree offset) will "exceed crosswind limitations," intercept and inject a metric correction stating the crosswind is mathematically negligible and acts as a safe headwind.
3. TAILWIND EVALUATION: If the relative wind angle to the target runway is greater than 90°, evaluate the tailwind component: Tailwind = Wind Speed * cos(Relative Angle). If the tailwind component exceeds 10–15 knots, inject an explicit **CRITICAL TAILWIND LIMIT EXCEEDED** warning, as this violates standard narrow-body aircraft structural limits."""


class WeatherAgent:
    def __init__(self, weather_engine, rag_agent):
        self.weather = weather_engine
        self.rag = rag_agent

    def fetch_realtime_metar(self, icao: str) -> Optional[str]:
        import urllib.request
        import urllib.error
        import json
        url = f"https://aviationweather.gov/api/data/metar?ids={icao.upper()}&format=json"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (AviationCopilot)'})
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))
                if data and isinstance(data, list) and len(data) > 0:
                    return data[0].get('rawOb', '')
        except Exception as e:
            logger.warning(f"Failed to fetch live METAR for {icao}: {e}")
        return None

    def run(self, query: str) -> str:
        # 1. TAF Forecast Detection
        if "TAF" in query.upper():
            metar_str = query
            icao_match = re.search(r"\b([A-Z]{4})\b", query.upper())
            icao = icao_match.group(1) if icao_match else "Unknown"
            region = "Unknown"
        # 2. SIGMET Advisory Detection
        elif any(x in query.upper() for x in ["SIGMET", "WSUS", "WCCL", "WSTU", "WV2", "WV3"]) or any(x in query.upper() for x in ["SEV TURB", "SEV ICE", "VOLCANIC ASH", "SQL TS"]):
            metar_str = query
            icao_match = re.search(r"\b([A-Z]{4})\b", query.upper())
            icao = icao_match.group(1) if icao_match else "ENROUTE"
            region = "Unknown"
        # 3. Standard METAR Detection (Default)
        else:
            metar_match = re.search(r"\b(?:\d{3}|VRB)\d{2}(?:G\d{2})?(?:KT|MPS)\b", query, re.I)
            metar_str = None
            icao = "Unknown"
            region = "Unknown"
            
            if metar_match:
                full_match = re.search(r"([A-Z]{4}\s+\d{6}Z\s+.*?(?:A\d{4}|Q\d{4}))", query, re.I)
                if full_match:
                    metar_str = full_match.group(1)
                else:
                    metar_str = query
            else:
                possible_codes = re.findall(r"\b([A-Z]{3,4})\b", query.upper())
                EXCLUDED = {"WHAT", "HOW", "THE", "FOR", "AND", "YOU", "ARE", "GET", "RUN", "WIND", "TEMP"}
                filtered = [c for c in possible_codes if c not in EXCLUDED]
                for code in filtered:
                    metar_str = self.fetch_realtime_metar(code)
                    if metar_str:
                        icao = code
                        break
        
        if not metar_str:
            return "Error: No METAR, TAF, or SIGMET report could be extracted or fetched for this query."
            
        if icao == "Unknown":
            icao_match = re.search(r"\b([A-Z]{4})\b", metar_str.upper())
            if icao_match:
                icao = icao_match.group(1)
            
        high_profile = {
            "KJFK": "US", "KLAX": "US", "KORD": "US", "KDFW": "US", "KDEN": "US",
            "EGLL": "UK", "LFPG": "FR", "EHAM": "NL", "EDDF": "DE", "RJTT": "JP"
        }
        if icao in high_profile:
            region = high_profile[icao]
        elif self.rag and icao != "ENROUTE":
            try:
                res = self.rag.airports_collection.get(where={"ident": icao})
                if res and res.get("metadatas") and res["metadatas"]:
                    region = res["metadatas"][0].get("country", "Unknown")
            except Exception:
                pass
                
        result = self.weather.predict(metar_str, region)
        
        # Add detailed temp/dewpoint spread warning
        spread_warning = ""
        features = result["features"]
        if "temperature" in features and "dewpoint" in features and features["dewpoint"] is not None:
            spread = features["temperature"] - features["dewpoint"]
            if spread <= 3.0:
                spread_warning = f" (Warning: Close temp-dewpoint spread of {spread:.1f}°C indicates high fog/mist potential)"
                
        # Format explicitly with units
        wind_gust_val = features.get("wind_gust")
        severe_icing_flag = features.get("severe_icing_alert", False)
        severe_turb_flag = features.get("severe_turbulence_alert", False)
        report_type = features.get("report_type", "METAR")
        
        formatted_features = {
            "wind_speed_knots": features.get("wind_speed", 0.0),
            "wind_gust_knots": wind_gust_val,
            "visibility_miles": features.get("visibility", 10.0),
            "temperature_celsius": features.get("temperature", 20.0),
            "dewpoint_celsius": features.get("dewpoint"),
            "ceiling_feet": features.get("ceiling", 99999.0),
            "severe_icing_alert": severe_icing_flag,
            "severe_turbulence_alert": severe_turb_flag,
            "report_type": report_type,
            "region": features.get("region", "Unknown")
        }

        # Create a detailed human-readable summary specifying standard aviation units and alerts
        gust_str = f", Gusts: {wind_gust_val} KT" if wind_gust_val is not None else ""
        alerts = []
        if severe_icing_flag:
            alerts.append("[ALERT] SEVERE ICING GROUND ALERT: ACTIVE (Freezing precipitation/fog in sub-zero temperature)")
        if severe_turb_flag:
            alerts.append("[ALERT] SEVERE EN-ROUTE TURBULENCE ADVISORY: ACTIVE")
        alerts_str = "\n".join([f"- {a}" for a in alerts])
        if alerts_str:
            alerts_str = "\n" + alerts_str

        summary_text = (
            f"Weather Observations for {icao} ({report_type}):\n"
            f"- Source Text: {metar_str}\n"
            f"- Wind Speed: {features.get('wind_speed', 0.0)} KT (knots){gust_str}\n"
            f"- Visibility: {features.get('visibility', 10.0):.2f} SM (statute miles)\n"
            f"- Temperature: {features.get('temperature', 20.0)}°C (celsius)\n"
            f"- Dewpoint: {features.get('dewpoint')}°C (celsius)\n"
            f"- Temperature-Dewpoint Spread: {result.get('temperature_dewpoint_spread')}°C{spread_warning}\n"
            f"- Ceiling: {features.get('ceiling', 99999.0)} feet (ft) above ground level\n"
            f"- Delay Risk Prediction: {result['risk']} (Confidence: {result['confidence'] * 100:.1f}%){alerts_str}"
        )

        out = {
            "retrieved_metar": metar_str,
            "parsed_features": formatted_features,
            "weather_summary": summary_text,
            "temperature_dewpoint_spread": result.get("temperature_dewpoint_spread"),
            "spread_warning": spread_warning,
            "severe_icing_alert": severe_icing_flag,
            "severe_turbulence_alert": severe_turb_flag,
            "risk_prediction": result["risk"],
            "confidence": result["confidence"],
            "airport_icao": icao,
            "region": region
        }
        return json.dumps(out, indent=2)


class RAGAgent:
    def __init__(self, rag_engine):
        self.rag = rag_engine
        self.collection = rag_engine.collection

    def run(self, query: str) -> str:
        return self.rag.retrieve(query)


class PerformanceAgent:
    def __init__(self, calculator):
        self.calculator = calculator

    def run(self, query: str) -> str:
        q = query.lower()
        if any(x in q for x in ["weight", "mtow", "payload", "cargo", "zfw", "takeoff weight"]):
            zfw_match = re.search(r"\b(?:zfw|zero\s*fuel\s*weight)\s*(?:of|is|=)?\s*(\d{2,6}(?:\.\d+)?)\b", q)
            fuel_match = re.search(r"\b(?:fuel|block\s*fuel)\s*(?:of|is|=)?\s*(\d{2,6}(?:\.\d+)?)\b", q)
            payload_match = re.search(r"\b(?:payload|cargo|passengers?)\s*(?:of|is|=)?\s*(\d{2,6}(?:\.\d+)?)\b", q)
            mtow_match = re.search(r"\b(?:mtow|max\s*takeoff\s*weight|limit)\s*(?:of|is|=)?\s*(\d{2,6}(?:\.\d+)?)\b", q)
            
            tow_explicit = None
            tow_match = re.search(r"\b(?:tow|takeoff\s*weight|calculated\s*weight)\s*(?:of|is|=)?\s*(\d{2,6}(?:\.\d+)?)\b", q)
            if tow_match:
                tow_explicit = float(tow_match.group(1))
                
            zfw = float(zfw_match.group(1)) if zfw_match else 58500.0
            fuel = float(fuel_match.group(1)) if fuel_match else 16000.0
            payload = float(payload_match.group(1)) if payload_match else 9000.0
            mtow = float(mtow_match.group(1)) if mtow_match else 79000.0
            
            if tow_explicit is not None:
                zfw = tow_explicit - fuel - payload
            else:
                all_nums = [float(x) for x in re.findall(r"\b(\d{5})\b", q)]
                if all_nums:
                    larger_nums = [n for n in all_nums if n > mtow]
                    if larger_nums:
                        tow_explicit = larger_nums[0]
                        zfw = tow_explicit - fuel - payload

            res = self.calculator.verify_weight_limits(zfw, fuel, payload, mtow)
            warning_str = f"\n{res['weight_warning']}" if res['weight_warning'] else ""
            return (
                f"Weight and Balance Performance Analysis:\n"
                f"| Parameter | Weight Value |\n"
                f"| --- | --- |\n"
                f"| Zero Fuel Weight (ZFW) | {res['zero_fuel_weight_kg']:.1f} kg |\n"
                f"| Fuel Weight | {res['fuel_weight_kg']:.1f} kg |\n"
                f"| Payload / Cargo | {res['payload_weight_kg']:.1f} kg |\n"
                f"| **Calculated Takeoff Weight (TOW)** | **{res['takeoff_weight_kg']:.1f} kg** |\n"
                f"| **Maximum Takeoff Weight (MTOW)** | **{res['max_takeoff_weight_kg']:.1f} kg** |\n"
                f"- Takeoff clearance status: {'APPROVED' if res['is_legal_takeoff'] else 'REJECTED - OVERWEIGHT'}{warning_str}"
            )
        elif any(x in q for x in ["landing", "runway", "stop", "landing distance"]):
            weight_match = re.search(r"\b(?:weight|wt|mass)\s*(?:of\s*)?(\d+(?:\.\d+)?)\b", q)
            weight = float(weight_match.group(1)) if weight_match else 60000.0
            
            wind_match = re.search(r"\b(?:wind|headwind|tailwind|speed)\s*(?:of\s*)?(\d+(?:\.\d+)?)\b", q)
            wind = float(wind_match.group(1)) if wind_match else 10.0
            
            calc_res = self.calculator.landing_distance(weight, wind)
            return (
                f"Landing Distance Calculation:\n"
                f"- Input Weight: {weight} kg\n"
                f"- Input Wind: {wind} KT\n"
                f"- Required Landing Distance: {calc_res['required_landing_distance_ft']} ft\n"
                f"- Formula used: 1500 + (weight * 0.8) - (wind * 15)"
            )
        else:
            dist_match = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:nm|nautical\s*miles?|mi|miles?|distance)\b", q)
            distance = float(dist_match.group(1)) if dist_match else 500.0
            
            burn_match = re.search(r"\b(?:burn\s*rate|flow|rate)\s*(?:of\s*)?(\d+(?:\.\d+)?)\b", q)
            burn_rate = float(burn_match.group(1)) if burn_match else 2600.0
            
            calc_res = self.calculator.fuel_burn(distance, burn_rate)
            return (
                f"Fuel Burn Calculation:\n"
                f"- Input Distance: {distance} NM\n"
                f"- Input Burn Rate: {burn_rate} kg/h\n"
                f"- Estimated Fuel Required: {calc_res['estimated_fuel_kg']} kg\n"
                f"- Formula used: (distance * burn_rate) / 60"
            )


class DecisionAgent:
    def __init__(self):
        self.weather_engine = weather_engine
        self.rag_engine = rag_engine
        self.calculator = FlightCalculator()
        
        self.rag_agent = RAGAgent(self.rag_engine)
        self.weather_agent = WeatherAgent(self.weather_engine, self.rag_agent)
        self.perf_agent = PerformanceAgent(self.calculator)
        
        self.llm = LocalLLM()

    def classify_intent(self, query: str) -> str:
        q = query.lower()
        # Programmatic keyword override for performance/calculator/weight queries
        if any(x in q for x in ["weight", "mtow", "payload", "cargo", "zfw", "takeoff weight", "landing distance", "fuel burn"]):
            return "CALCULATOR"
            
        has_metar = bool(re.search(r"\b(?:\d{3}|VRB)\d{2}(?:G\d{2})?(?:KT|MPS)\b", query, re.I))
        
        has_weather_keywords = any(x in q for x in ["metar", "weather", "delay", "visibility", "wind", "forecast", "ceiling", "dewpoint", "spread"])
        
        possible_codes = re.findall(r"\b([A-Z]{3,4})\b", query.upper())
        EXCLUDED = {"WHAT", "HOW", "THE", "FOR", "AND", "YOU", "ARE", "GET", "RUN"}
        has_airport = any(c not in EXCLUDED for c in possible_codes)
        
        if has_metar or (has_weather_keywords and has_airport):
            return "WEATHER"
            
        if "calculate" in q or "calculator" in q:
            return "CALCULATOR"
            
        if any(x in q for x in [
            "checklist", "engine", "procedure", "limitation", "manual", "system", 
            "autopilot", "speed", "altitude", "landing", "takeoff", "weight", 
            "emergency", "airport", "code", "icao", "iata"
        ]):
            return "RAG"
            
        if any(x in q for x in ["fuel", "burn", "distance", "runway"]):
            return "CALCULATOR"
            
        if has_metar:
            return "WEATHER"
            
        return "GENERAL"

    def run(self, query: str) -> str:
        intent = self.classify_intent(query)
        logger.info(f"DecisionAgent routed query to: {intent}")
        
        system_instruction = "You are an Aviation Flight Operations Copilot."
        if intent == "WEATHER":
            system_instruction = WEATHER_AGENT_SYSTEM_PROMPT
            information = self.weather_agent.run(query)
        elif intent == "RAG":
            system_instruction = "You are an Aviation Flight Operations Copilot querying the flight crew operational manuals."
            information = self.rag_agent.run(query)
        elif intent == "CALCULATOR":
            system_instruction = PERFORMANCE_AGENT_SYSTEM_PROMPT
            information = self.perf_agent.run(query)
        else:
            information = "No specialized tool selected."
            
        final_prompt = f"""
System Role Instructions:
{system_instruction}

Use the operational information below. Always cite the specific source document and page/lookup details (e.g. [Source: airports.csv], [Source: B738-FCOM-Rev21-15Sep2016.pdf, Page: 45], [Source: METAR]) when using information retrieved from the manuals or databases.

Always provide:
- Safety considerations
- Operational recommendation
- Limitations

Operational Principles to Follow:
- Severe Icing Alert: If the weather information contains a 'SEVERE ICING GROUND ALERT', highlight it immediately under Safety Considerations. Emphasize that freezing precipitation or freezing fog in sub-zero temperatures represents an immediate ground safety hazard requiring pre-takeoff de-icing/anti-icing.
- Severe Turbulence Advisory: If the weather information contains a 'SEVERE EN-ROUTE TURBULENCE ADVISORY', highlight it immediately under Safety Considerations. Emphasize that severe turbulence threatens flight control and passenger safety, requiring altitude changes or holding patterns.
- Wind Gusts and Crosswinds: Always check for wind gusts (e.g. G30KT) in the weather observations. Under Safety Considerations, evaluate how these wind gusts might result in severe crosswinds relative to the destination runway orientation (e.g., a 340° wind at KJFK is a strong crosswind on Runway 04/22).
- RVR Physical Definition: Runway Visual Range (RVR, e.g. R04R/2000V4000FT) is strictly a measure of horizontal visibility down the runway centerline. It does NOT indicate or affect runway friction, water depth, or braking action. Do not state that low RVR causes hydroplaning or reduced stopping distance; only precipitation types (like snow, freezing rain) or runway condition states—not RVR—indicate braking degradation.

User Query:
{query}

Available Information:
{information}
"""
        draft = self.llm.generate(final_prompt)
        
        # Quality control auditing using Centralized ValidatorAgent rules
        validator_prompt = f"""
{VALIDATOR_AGENT_SYSTEM_PROMPT}

Audit the combined output draft below against the raw telemetry data. Ensure zero-tolerance for conflating low RVR values with braking degradation, check for mathematically accurate crosswinds, and flag structural tailwind violations.

Raw Telemetry/Input:
Query: {query}
Specialist Output: {information}

Draft to Audit:
{draft}

Generate the final validated, safe, and audited briefing output:
"""
        return self.llm.generate(validator_prompt)


class OpsCopilotAgent(DecisionAgent):
    pass






# ============================================================
# SECTION 8
# COMMAND LINE INTERFACE
# ============================================================



def start_cli():


    agent = OpsCopilotAgent()



    console.print(

        """

================================================

       AI FLIGHT OPERATIONS COPILOT

 Local Aviation Intelligence Assistant

================================================

Type 'exit' to quit.

"""

    )



    while True:


        try:


            query = console.input(

                "\n[bold green]Dispatcher > [/bold green]"

            )


            if query.lower()=="exit":

                break



            response = agent.run(
                query
            )



            console.print(

                "\n[bold cyan]COPILOT:[/bold cyan]"

            )


            console.print(
                response
            )



        except Exception as exc:


            logger.exception(exc)

            console.print(

                "[red]System error occurred[/red]"

            )







# ============================================================
# APPLICATION ENTRY POINT
# ============================================================



if __name__=="__main__":

    start_cli()