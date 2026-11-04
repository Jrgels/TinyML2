// TinyML 2026 | Aproximacion de sin(x) con un modelo INT8 propio.
// Placas objetivo: Arduino Nano 33 BLE Sense / Sense Rev2 (nRF52840).
// Biblioteca: Chirale_TensorFlowLite 2.0.0.
#include <Arduino.h>
#include <math.h>
#include <stdint.h>
#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"   // se modificó para usar MicroMutableOpResolver
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model.h"

// Variables globales
constexpr int kTensorArenaSize = 24 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* net = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

// Medición de tiempos
const int NUM_PRUEBAS = 100;          // inferencias por entrada
unsigned long tiempos[NUM_PRUEBAS];

const int NUM_CASOS = 5;
const float entradas[NUM_CASOS] = {
  0.0f,
  1.57f,   // pi/2
  3.14f,   // pi
  4.71f,   // 3pi/2
  6.28f    // 2pi
};

bool medicion_realizada = false;

void fatal(const char* message) {
  Serial.println(message);
  while (true) { delay(1000); }
}

void medirCaso(float x) {
  // Cuantización
  const float s_in = input_tensor->params.scale;
  const int z_in = input_tensor->params.zero_point;
  int qx = (int)lroundf(x / s_in) + z_in;
  if (qx < -128) qx = -128;
  if (qx >  127) qx =  127;
  const int8_t qx8 = (int8_t)qx;

  // Calentamiento y lectura de la salida
  input_tensor->data.int8[0] = qx8;
  if (interpreter->Invoke() != kTfLiteOk) {
    fatal("ERROR: fallo de inferencia (calentamiento).");
  }
  int8_t qy = output_tensor->data.int8[0];

  // Región cronometrada
  for (int i = 0; i < NUM_PRUEBAS; i++) {
    input_tensor->data.int8[0] = qx8;
    unsigned long inicio = micros();
    if (interpreter->Invoke() != kTfLiteOk) {
      fatal("ERROR: fallo de inferencia.");
    }
    unsigned long fin = micros();
    tiempos[i] = fin - inicio;
  }

  // Estadísticos
  unsigned long minimo = tiempos[0];
  unsigned long maximo = tiempos[0];
  double suma = 0;
  for (int i = 0; i < NUM_PRUEBAS; i++) {
    if (tiempos[i] < minimo) minimo = tiempos[i];
    if (tiempos[i] > maximo) maximo = tiempos[i];
    suma += tiempos[i];
  }
  double promedio = suma / NUM_PRUEBAS;

  double suma_cuadrados = 0;
  for (int i = 0; i < NUM_PRUEBAS; i++) {
    double d = (double)tiempos[i] - promedio;
    suma_cuadrados += d * d;
  }
  double desviacion = sqrt(suma_cuadrados / NUM_PRUEBAS);

  float y = (float)(qy - output_tensor->params.zero_point)
            * output_tensor->params.scale;
  float real = sinf(x);

  // --- Resultados
  Serial.println("----------------------------------");
  Serial.print("Entrada x = "); Serial.println(x, 4);
  Serial.print("sin(x) real = "); Serial.print(real, 4);
  Serial.print(" | prediccion = "); Serial.println(y, 4);

  Serial.print("Minimo: ");   Serial.print(minimo);
  Serial.print(" us (");      Serial.print(minimo / 1000.0, 3);   Serial.println(" ms)");

  Serial.print("Maximo: ");   Serial.print(maximo);
  Serial.print(" us (");      Serial.print(maximo / 1000.0, 3);   Serial.println(" ms)");

  Serial.print("Promedio: "); Serial.print(promedio, 2);
  Serial.print(" us (");      Serial.print(promedio / 1000.0, 3); Serial.println(" ms)");

  Serial.print("Desv. estandar: "); Serial.print(desviacion, 2);
  Serial.print(" us (");            Serial.print(desviacion / 1000.0, 3); Serial.println(" ms)");
}

void setup() {
  Serial.begin(115200);
  unsigned long inicio = millis();
  while (!Serial && millis() - inicio < 3000) { delay(10); }

  net = tflite::GetModel(g_model);
  if (net->version() != TFLITE_SCHEMA_VERSION) {
    fatal("ERROR: version de schema incompatible.");
  }

  // N = 2 tipos de operador (FULLY_CONNECTED y TANH)
  static tflite::MicroMutableOpResolver<2> resolver;
  if (resolver.AddFullyConnected() != kTfLiteOk) {
    fatal("ERROR: no se pudo registrar FULLY_CONNECTED.");
  }
  if (resolver.AddTanh() != kTfLiteOk) {
    fatal("ERROR: no se pudo registrar TANH.");
  }

  static tflite::MicroInterpreter static_interpreter(
      net, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    fatal("ERROR: AllocateTensors. Revisar arena/ops/versiones.");
  }
  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);
  if (input_tensor->type != kTfLiteInt8 || output_tensor->type != kTfLiteInt8) {
    fatal("ERROR: se esperaba entrada y salida INT8.");
  }
  if (input_tensor->params.scale <= 0 || output_tensor->params.scale <= 0) {
    fatal("ERROR: escalas de cuantizacion invalidas.");
  }
  Serial.println("Inicializacion completa (MicroMutableOpResolver<2>).");
}

void loop() {
  if (medicion_realizada) return;
  medicion_realizada = true;

  Serial.println("RESULTADOS DE RENDIMIENTO (100 inferencias por entrada)");
  for (int c = 0; c < NUM_CASOS; c++) {
    medirCaso(entradas[c]);
  }
  Serial.println("----------------------------------");
  Serial.println("Medicion terminada.");
}