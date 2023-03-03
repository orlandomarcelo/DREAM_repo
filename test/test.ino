int ledPin = 9;

void setup() {
  pinMode(ledPin, OUTPUT);
  Serial.begin(9600);
}

void loop() {
  if (Serial.available()) {
    int pwmVal = Serial.parseInt();
    analogWrite(ledPin, pwmVal);
  }
}