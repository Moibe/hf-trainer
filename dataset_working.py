from datasets import load_dataset

#trust_remote se usa actualmente para confiar en los datasets que tienen código incluído.
#Será mandatorio para el siguiente realese grande de datasets.
emotions_dataset = load_dataset("emotion", trust_remote_code=True)

# print(emotions_dataset)
# print(emotions_dataset["train"])

emotion_df = emotions_dataset["train"].to_pandas()
# print(emotions_df.head())

features = emotions_dataset["train"].features

print(features["label"])

#print(features["label"].int2str(0))

#Poner etiqueta textual a los labels numéricos para que sean más explícitas.
id2label = {idx:features["label"].int2str(idx) for idx in range(6)}

print(id2label)

label2id = {v:k for k,v in id2label.items()}

print(label2id)

#Investigar la distribución de las clases.
distribucion = emotion_df["label"].value_counts(normalize=True).sort_index()
print(distribucion)

#Si tiene una mala distribución, es decir por ejemplo, muchos ejemplos de sadness, significará que el modelo se volverá bueno para precedir sadness perp muy malo para...
#predecir por ejemplo sorpresa, que solo tiene el 3% de los ejemplos.