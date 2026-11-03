En este repositorio se creará el proyecto 2 de la materia de Tiny Macine Learning, el cuál consiste de lo siguiente:

1. Descripción del proyecto
En esta práctica se desarrollará e implementará una red neuronal artificial capaz de aproximar la función matemática:

y=sin(x)

El propósito es experimentar con el ciclo completo de desarrollo de una aplicación TinyML, desde la generación de datos y el entrenamiento de una red neuronal hasta su optimización y ejecución en un microcontrolador con recursos limitados.

El entrenamiento se realizará utilizando TensorFlow y Keras en un entorno Python. Posteriormente, el modelo se convertirá al formato TensorFlow Lite, basado en FlatBuffers, y se aplicarán diferentes estrategias de cuantización con el propósito de reducir sus requerimientos de almacenamiento y procesamiento.

Se analizarán las diferencias entre las representaciones de punto flotante de 32 bits (FP32), punto flotante de 16 bits (FP16) y enteros de 8 bits (INT8), considerando tanto el tamaño de los modelos como su precisión y estructura interna.

Los modelos compatibles se exportarán como arreglos de bytes en lenguaje C/C++ y se integrarán en programas desarrollados mediante Arduino IDE para ejecutarlos en la tarjeta Arduino Nano 33 BLE Sense.

Se realizarán experimentos para evaluar la precisión de los modelos, los recursos de memoria utilizados y el tiempo de inferencia. También se analizará el efecto de seleccionar únicamente los operadores necesarios de TensorFlow Lite Micro, la influencia del conjunto representativo utilizado durante la cuantización y el comportamiento de los modelos cuando reciben entradas fuera del intervalo utilizado durante su entrenamiento.

Esta práctica permitirá comprender las diferencias entre entrenar una red neuronal en un entorno de cómputo convencional y ejecutar inferencias en un sistema embebido, así como los compromisos entre precisión, tamaño del modelo, consumo de memoria y rendimiento computacional.

2. Objetivo general
Desarrollar, cuantizar, optimizar e implementar una red neuronal artificial en un microcontrolador Arduino Nano 33 BLE Sense mediante TensorFlow Lite Micro, evaluando la influencia de diferentes estrategias de cuantización sobre la representación, precisión, tamaño y rendimiento del modelo, para comprender los principios fundamentales del aprendizaje automático en dispositivos con recursos limitados (TinyML).

2.1. Objetivos específicos
Diseñar y entrenar una red neuronal para aproximar una función matemática continua utilizando TensorFlow/Keras.

Convertir el modelo entrenado al formato TensorFlow Lite y analizar su estructura interna basada en FlatBuffers.

Aplicar diferentes estrategias de cuantización y comparar sus efectos sobre el tamaño, la representación numérica y la precisión del modelo.

Comprender el funcionamiento de la cuantización y el papel de los parámetros scale y zero_point.

Exportar modelos cuantizados e integrarlos en una aplicación para Arduino.

Implementar la ejecución de inferencias mediante TensorFlow Lite Micro.

Medir los requerimientos de memoria Flash, RAM y el tiempo de inferencia.

Optimizar la aplicación mediante la selección de los operadores requeridos por la red neuronal.

Analizar la capacidad de generalización del modelo ante entradas dentro y fuera del intervalo de entrenamiento.

Evaluar la influencia del conjunto representativo de calibración sobre el comportamiento de los modelos cuantizados.

Identificar los compromisos entre precisión, tamaño, memoria y rendimiento que intervienen en el despliegue de modelos TinyML.