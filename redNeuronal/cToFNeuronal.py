import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, -14, 32, 46, 59, 72, 100], dtype=float)

capa = tf.keras.layers.Dense(units=1, input_shape=[1]) #Las capas densas todas las neuronas de la capa tienen una conexion con todas las neuronas de la siguiente capa
modelo = tf.keras.Sequential([capa])

modelo.compile( ##compila el modelo para ser entrenado
    optimizer = tf.keras.optimizers.Adam(0.1), #Adam permite ajustar los pesos y sesgos para que aprenda y no empeore, el parametro sera la tasa de aprendizaje con la cual modifica los valores
    loss = "mean_squared_error" #una poca cantidad de errores grandes es peor que una gran cantidad de errors pequeños
)

print("Empazando entrenamiento...")
historial = modelo.fit(celsius, fahrenheit, epochs=1000, verbose=False) #le damos valores suministrados y valores esperados, y la cantidad de veces que lo va a intentar
print('Modelo Entrenado!!')

plt.xlabel('#Epoca')
plt.ylabel("Magnitud de perdida")
plt.plot(historial.history["loss"])

print("Hagamos una prediccion!!!")
resultado = modelo.predict([30.0])
print(f"El resultado es {str(resultado)} F\n")

print("variables internas del modelo:")
print(capa.get_weights())

plt.show()