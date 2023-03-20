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

// this constant won't change:
const int  flashPin = 4;    // the pin that the pushbutton is attached to
const int controlPin = 5;       // the pin that the LED is attached to
const int ledPin = 2;
const int nbBaseline = 2;
const int totalPoint = 17;

int delais_LED = 40;




// Variables will change:
int buttonState = 0;         // current state of the button
int countFlashBaseline = 0;

void setup() {
  // initialize the button pin as a input:
  pinMode(flashPin, INPUT);
  // initialize the LED as an output:
  pinMode(controlPin, OUTPUT);
  digitalWrite(controlPin,HIGH);

  pinMode(ledPin, OUTPUT);
  digitalWrite(ledPin,LOW);
  // initialize serial communication:
  Serial.begin(9600);
}


void loop() {

    // read the pushbutton input pin:
  buttonState = digitalRead(flashPin);
  if (buttonState == 0){
      digitalWrite(ledPin,HIGH);
      delay(delais_LED);
      digitalWrite(ledPin,LOW);
    }
}
