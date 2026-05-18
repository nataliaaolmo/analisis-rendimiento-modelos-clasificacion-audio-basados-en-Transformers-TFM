from transformers import AutoModelForAudioClassification

model_path = "./hubert-base-ls960-local-finetuned1"  

model = AutoModelForAudioClassification.from_pretrained(model_path)

total_params = sum(p.numel() for p in model.parameters())

print(f"Parámetros totales:     {total_params:,}")
#cada parámetro de una red neuronal es un número en coma flotante de 32 bits = 4 bytes, lo multiplica primero *4 para pasarlo a bytes; y 1024**2 hace dos divisiones de golpe, porque hay que dividirlo entre 1024 para pasarlo a kilobytes y otra vez para pasarlo a megabytes.
print(f"Tamaño aprox float32:   {total_params * 4 / 1024**2:.1f} MB") 