import os
import pandas as pd
import time
import google.generativeai as genai
import pandas as pd

# from google.colab import userdata
# GOOGLE_API_KEY=userdata.get('GEMINI_API_KEY')
from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

def main(temperature = 1):
    generation_config = {
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    safety_settings = [
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE"
        },
    ]

    system_instruction = """
    Humanize the below given 10 prompts by converting them into basic, simple, and concise one-liners.\n
    Keep the context and important keyword but use human-like language, including possible spelling mistakes and typos. 
    The output should be less descriptive and shorter in length and should not contain technical terms except for basic keywords. 
    Remember to just list the prompts only don't give any.
    """

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        safety_settings=safety_settings,
        system_instruction=system_instruction
    )

    advanced_df = pd.read_excel("500_technical_prompts_dataset.xlsx", header=None)
    chunk_size = 10
    combined_rows = []

    for i in range(0, len(advanced_df), chunk_size):
        chunk = advanced_df.iloc[i:i+chunk_size]
        combined_row = "\n".join([f"{j+1}. {str(x)}" for j, x in enumerate(chunk.values.tolist())])
        combined_rows.append(combined_row)

    def convert_prompts(advanced_prompt):
        chat_session = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [
                        "1. ['Illustration, Eerie, an Environmental art of 1woman of 25yo, neon green hair, sulking, Ruined Kingdom of France']\\n2. ['sketch, woman wearing Pinafore dress and ruffled blouse, masterpiece, 8k, high resolution, shallow depth of field, sharp focus']\\n3. ['Neon Cyborg, abstract contemporary, feminism, installation, mixed-media, organic Metal particles and pieces in the air Envision an ancient robotic humanshaped Computer standing in a vast, sun-kissed prairie. The boundless expanse stretches to the horizon, and the robotic human graceful figure harmonizes with the natural beauty of the open wasteland. Pencil sketch, sketch']\\n4. ['(masterpiece, best quality:1.1), (sketch:1.1), paper, no humans, (a rose:1.1), flower, stem, thorn, leaf, plant']\\n5. ['dynamic shot of a beautiful red ruby and (diamonds:1.2) scaled dragon|cat hybrid, (beautiful eyes, up close:1.3), macro, mythological creature, dream world, pencil sketch, pencil drawing, monochrome, highly detailed']\\n6. ['((masterpiece,best quality, detailed)), 1boy, male focus, (sketch:1.1), paper, monochrome, cropped torso, emotionless, looking at viewer, white shirt']\\n7. ['a black fat man, a straw hat, (farmer, dirty clothes:1.1), a field with wheat in the background, a ranch, warm lighting, cozy atmosphere']\\n8. ['by Matias Hannecke, (electrifying but extremely beautiful:1.4), (intricate details, masterpiece, best quality:1.4), in the style of Nicola Samori, Futuristic style, sleek, modern, ultramodern, cartoon']\\n9. ['(18yo redhead girl:1.2), makeup, graphic eyeliner, rouge, (choker:0.9), realistic skin texture, oversize knit sweater, (red:0.8)']\\n10. ['head tilt, upper body, pink hair, dress, (best quality, masterpiece ,ultra-detailed:1.1), 1girl, solo, looking at viewer, smile, 3d illustration']",
                    ],
                },
                {
                    "role": "model",
                    "parts": [
                        "Here are the prompts, humanized and shortened:\n\n1.  Eerie pic of a girl with green hair, sad, France is ruined.\n2.  Sketch of a woman in a fancy dress, super detailed.\n3.  Neon robot lady in a field, pencil sketch.\n4.  Sketch of a rose, no people.\n5.  Dragon-cat with a ruby, close up, pencil sketch.\n6.  Sketch of a boy, white shirt, looking at you.\n7.  Black man, straw hat, farmer, wheat field.\n8.  Cartoon in Nicola Samori style, futuristic, super detailed.\n9.  Girl with red hair, makeup, big sweater.\n10.  Girl with pink hair, smiling, looking at you, 3D. \n",
                    ],
                },
            ]
        )
        try:
            response = chat_session.send_message(advanced_prompt)
            time.sleep(5) 
            return response.text
        except Exception as e:
            print(f"Error occurred: {e}. Retrying...")

    def get_formatted_prompts(combined_row):
        while True:
            response = convert_prompts(combined_row)
            lines = response.split("\n")
            formatted_prompts = []
            for line in lines[2:]:
                if line.strip():
                    formatted_prompts.append(line.lstrip("0123456789. ").strip())
            if len(formatted_prompts) >= 10:
                return formatted_prompts

    results = []
    for i, combined_row in enumerate(combined_rows):
        print(f"Iteration: {i+1}")
        results.append(get_formatted_prompts(combined_row))

    prompts_df = pd.DataFrame({"human_prompts": [item for sublist in results for item in sublist]})
    excel_df = pd.DataFrame({"advanced_prompt": advanced_df.iloc[:, 0]})
    result_df = pd.concat([prompts_df, excel_df], axis=1)
    result_df.to_excel(f"500prompts{temperature}.xlsx", index=False)

if __name__ == "__main__":
    main(temperature=0.5)