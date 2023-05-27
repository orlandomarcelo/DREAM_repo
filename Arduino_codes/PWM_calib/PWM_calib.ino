// the number of the LED pin
const int ledPin = 2;

// the number of the flash pin
const int  flashPin = 5;

// setting PWM properties
const int freq = 1000 ; //refresh rate
const int ledChannel = 0;
const int resolution = 16;
int max_amp = pow(2, resolution) - 1;

float amp_fact = 0;

int PWM_value = 0;
int buttonState = 0;


void setup() {
  pinMode(flashPin, INPUT);
  // configure LED PWM functionalitites
  ledcSetup(ledChannel, freq, resolution);
  
  // attach the channel to the GPIO to be controlled
  ledcAttachPin(ledPin, ledChannel);
}

void loop() {
  if (buttonState == 0){
    amp_fact = 0.85;
    while (amp_fact <= 1){
      PWM_value = round(amp_fact * max_amp);
      amp_fact += 0.005;
      ledcWrite(ledChannel, PWM_value);
      delay(10000);
    }
    buttonState = 1;
    ledcWrite(ledChannel, 0);
  }
  
}
