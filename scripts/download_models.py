import asyncio
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_model(model_type: str, model_name: str, output_dir: str = "./models"):
    logger.info(f"Downloading {model_type}/{model_name} to {output_dir}")

    output_path = Path(output_dir) / model_type / model_name
    output_path.mkdir(parents=True, exist_ok=True)

    if model_type == "asr":
        download_asr_model(model_name, str(output_path))
    elif model_type == "llm":
        download_llm_model(model_name, str(output_path))
    elif model_type == "tts":
        download_tts_model(model_name, str(output_path))
    else:
        logger.error(f"Unknown model type: {model_type}")


def download_asr_model(model_name: str, output_path: str):
    logger.info(f"Downloading ASR model: {model_name}")

    try:
        from modelscope import snapshot_download
        cache_dir = snapshot_download(
            model_id=f"damo/speech_{model_name}",
            cache_dir=output_path
        )
        logger.info(f"ASR model downloaded to: {cache_dir}")
    except ImportError:
        logger.warning("modelscope not installed. Install with: pip install modelscope")
    except Exception as e:
        logger.error(f"Failed to download ASR model: {e}")


def download_llm_model(model_name: str, output_path: str):
    logger.info(f"Downloading LLM model: {model_name}")

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        tokenizer.save_pretrained(output_path)

        logger.info(f"LLM model downloaded to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to download LLM model: {e}")


def download_tts_model(model_name: str, output_path: str):
    logger.info(f"Downloading TTS model: {model_name}")
    logger.warning(f"TTS model download not implemented for: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Download AI models")
    parser.add_argument("--type", "-t", required=True, choices=["asr", "llm", "tts"])
    parser.add_argument("--name", "-n", required=True)
    parser.add_argument("--output", "-o", default="./models")

    args = parser.parse_args()
    download_model(args.type, args.name, args.output)


if __name__ == "__main__":
    main()
