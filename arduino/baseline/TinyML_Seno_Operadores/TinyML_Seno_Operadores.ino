// TinyML 2026 | Aproximacion de sin(x) con un modelo INT8 propio.
// Placas objetivo: Arduino Nano 33 BLE Sense / Sense Rev2 (nRF52840).
// Biblioteca: Chirale_TensorFlowLite 2.0.0.
#include <Arduino.h>
#include <math.h>
#include <stdint.h>
#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model.h"

constexpr int kTensorArenaSize = 24 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* net = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

void fatal(const char* message) {
  Serial.println(message);
  while (true) { delay(1000); }
}

void setup() {
  Serial.begin(115200);
  unsigned long inicio = millis();
  while (!Serial && millis() - inicio < 3000) { delay(10); }

  net = tflite::GetModel(g_model);
  if (net->version() != TFLITE_SCHEMA_VERSION) {
    fatal("ERROR: version de schema incompatible.");
  }

  //para saber operadores que usa el modelo
  auto* subgraph = (*net->subgraphs())[0];
  auto* codes = net->operator_codes();
  for (unsigned i = 0; i < subgraph->operators()->size(); i++) {
    auto* code = (*codes)[(*subgraph->operators())[i]->opcode_index()];
    Serial.print("Op "); Serial.print(i); Serial.print(": ");
    Serial.println(tflite::EnumNameBuiltinOperator(
      tflite::BuiltinOperator(code->builtin_code())));
  }


  static tflite::AllOpsResolver resolver;
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
  Serial.println("Inicializacion completa; abrir Herramientas > Serial Plotter.");
}

void loop() {
  static uint16_t paso = 0;
  constexpr uint16_t kPasos = 128;
  constexpr float kDosPi = 6.28318530718f;
  float x = kDosPi * (float)(paso % kPasos) / (float)(kPasos - 1);
  paso++;

  // Cuantizacion: q = round(x / scale) + zero_point, saturada a [-128,127].
  const float s_in = input_tensor->params.scale;
  const int z_in = input_tensor->params.zero_point;
  int qx = (int)lroundf(x / s_in) + z_in;
  if (qx < -128) qx = -128;
  if (qx >  127) qx =  127;
  input_tensor->data.int8[0] = (int8_t)qx;

  if (interpreter->Invoke() != kTfLiteOk) {
    fatal("ERROR: fallo de inferencia.");
  }

  int8_t qy = output_tensor->data.int8[0];
  float y = (float)(qy - output_tensor->params.zero_point)
            * output_tensor->params.scale;
  float real = sinf(x);

  // Salida apta para Arduino IDE Serial Plotter.
  //.print("real:"); Serial.print(real, 4);
  //Serial.print(",pred:"); Serial.println(y, 4);
  //delay(80);
}