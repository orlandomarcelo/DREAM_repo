const int potPin = 34;
const int flashPin = 5;

// variable for storing the potentiometer value
int potValue = 0;
int freq = 10000; // refresh rate
float max_time = 10; // Experiment time in seconds

unsigned long t = 0;
int PWM_value = 0;
unsigned long startMillis = 0;
int trigger = 0;

void setup() {
  max_time = max_time * 1000;
  pinMode(flashPin, INPUT);
  pinMode(potPin, INPUT);
  Serial.begin(115200);
  Serial.print("Time");
  Serial.print(";");
  Serial.print("Read");
  //Serial.println();
}

void loop() {
  trigger = digitalRead(flashPin);
  if (trigger == 1){
    startMillis = millis();
    t = 0;
    while (t < max_time ){
      potValue = analogRead(potPin);
      t = (millis() - startMillis);
      Serial.print(t);
      Serial.print(";");
      Serial.print(potValue);
      //Serial.println();
    }
    Serial.println("END");  
  }
}
