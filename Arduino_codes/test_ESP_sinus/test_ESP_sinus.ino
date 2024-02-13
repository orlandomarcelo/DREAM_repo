#include <math.h>

#define PI 3.1415926535

// the number of the LED pin
const int ledPin = 2;

// the number of the flash pin
const int  flashPin = 5;

// setting PWM properties
const int freq = 10000 ; //refresh rate
const int ledChannel = 0;
const int resolution = 13;
int max_amp = pow(2, resolution) - 1;

// setting the actinic light frequency or period
//float period = 16; // s
float frequency = 8; // Hz
//float frequency = 1/period;

// setting sine wave resolution
int Ts = //sampling period for the sine
int nb_points = 

// setting actinic light intensity
float offset_fact = 0.2; // Offset as a fraction of max intensity
float amp_fact =  0.1; // Amplitude of modulation a fraction of max intensity

// setting total time of the oscillations
float max_time = 650; // Experiment time in seconds

//IMPORTANT: 
// (amp_fact + offset_fact <= 1) and (amp_fact <= offset_fact=)

unsigned long t = 0;
int PWM_value = 0;
unsigned long startMillis = 0;
int trigger = 1;


void setup() {
  Serial.begin(115200);
  pinMode(flashPin, INPUT);
  max_time = max_time * 1000;
  // configure LED PWM functionalitites
  ledcSetup(ledChannel, freq, resolution);
  
  // attach the channel to the GPIO to be controlled
  ledcAttachPin(ledPin, ledChannel);

  // create signal (lookup table)


}

void loop() {
  ledcWrite(ledChannel, round(offset_fact * max_amp));
  trigger = digitalRead(flashPin);
  trigger = 1;
  if (trigger == 1){
    startMillis = millis();
    t = 0;
    while (t < max_time ){
      //Serial.println(trigger);
      t = (millis() - startMillis);
      PWM_value = round((amp_fact * max_amp * sin(2 * PI *(frequency/1000)* t) + offset_fact * max_amp));
      ledcWrite(ledChannel, PWM_value);
      Serial.println(PWM_value);
      //delayMicroseconds(1/freq);
      
    }
    ledcWrite(ledChannel, round(offset_fact * max_amp));
  }
}
