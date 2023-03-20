
/*
  State change detection (edge detection)

  Often, you don't need to know the state of a digital input all the time, but
  you just need to know when the input changes from one state to another.
  For example, you want to know when a button goes from OFF to ON. This is called
  state change detection, or edge detection.

  This example shows how to detect when a button or button changes from off to on
  and on to off.

  The circuit:
  - pushbutton attached to pin 2 from +5V
  - 10 kilohm resistor attached to pin 2 from ground
  - LED attached from pin 13 to ground (or use the built-in LED on most
    Arduino boards)

  created  27 Sep 2005
  modified 30 Aug 2011
  by Tom Igoe

  This example code is in the public domain.

  http://www.arduino.cc/en/Tutorial/ButtonStateChange
*/
int periode = 244; //2046 valeur
int nb_periode = 30;



const int  flashPin = 4;
const int ledPin = 2;
int i_dutyCycle = 0;
int buttonState = 0; 


// Paramètre du channel 0 du PWM
const int freq = 40000; // 5000 Hz
const int ledChannel = 0;
const int PWMResolution = 12; // Résolution de 14 bits
const int  Max_duty_cycle = pow(2, PWMResolution) - 1;

void setup(){
    // Configure le channel 0
    ledcSetup(ledChannel, freq, PWMResolution);

    // Attache le channel 0 sur les 3 pins
    ledcAttachPin(ledPin, ledChannel);
}

void loop(){
  //buttonState = digitalRead(flashPin);
  if (buttonState == 0){
    for(int i=0; i<nb_periode; i++){
        // Augmente la luminosité de la led
        while(i_dutyCycle < Max_duty_cycle)
      {
        ledcWrite(ledChannel, i_dutyCycle++);
        delayMicroseconds(periode);
      }
      while(i_dutyCycle > 0)
      {
        ledcWrite(ledChannel, i_dutyCycle--);
        delayMicroseconds(periode);
    }
   }
  }
}
