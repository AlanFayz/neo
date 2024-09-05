package main

import (
    "fmt"
    "os"
	"strings"
	"encoding/json"
	"regexp"
	"bufio"
	"time"

)


type Tokens struct {
    Tokens []string
	path string
	saved bool
}


func check(e error) {
    if e != nil {
        panic(e)
    }
}


func (x *Tokens) NewToken(t string){
	x.Tokens = append(x.Tokens,t)
	x.saved = false
}

func (x *Tokens) SaveTokens(){
	if x.saved{
		return
	}
	x.saved = true
	data, _ := json.Marshal(x)
	err := os.WriteFile(x.path, data,0644)	
	check(err)

}


func (x *Tokens) CommitToken(t string) int {
	x.NewToken(t)
	x.SaveTokens()
	return len(x.Tokens)-1
}



func LoadTokens(file string) Tokens{
	dat, err := os.ReadFile(file)
	check(err)
	var tokens Tokens
    er := json.Unmarshal(dat, &tokens)
	check(er)
	tokens.path= file
	tokens.saved = true
	return tokens

}

func removePunctuation(str string) string {
    punctuationRegex := regexp.MustCompile(`[[:punct:]]`)

    return punctuationRegex.ReplaceAllString(str, "")
}
func (x *Tokens) GetToken(token string,flags ...bool) int{
	create := false
	save := false
	if len(flags) > 0 {
    	create = flags[0]
  	}
	if len(flags) > 1 {
		save = flags[1]
	}

	token = strings.ToLower(token)
	token = removePunctuation(token)
	for ind, tok := range x.Tokens{
		if tok == token{
			return ind
		}
	}
	t := -1
	if create{
		x.NewToken(token)
		t = len(x.Tokens)-1
	}
	if save{
		x.SaveTokens()
	}
	return t



}


type DataInput struct{
	Data []string
} 


type DataSet struct{
	Data map[string][]int
}


func LoadDataInput(file string) DataInput {
	dat, err := os.ReadFile(file)
	check(err)
	var data DataInput
    er := json.Unmarshal(dat, &data)
	check(er)
	return data

}


func (x *Tokens) TokenizeSentence(inp string, flags ...bool) []int{
	res := strings.Split(inp, " ")
	tokens := []int{}
	for _, t := range res {
		tokens = append(tokens,x.GetToken(t,flags...))

	}
	return tokens
}





func CreateOutput(inp DataInput,x *Tokens,output string){
	ds := DataSet{}
	ds.Data = make(map[string][]int)


	for _, d := range inp.Data{
		ds.Data[d] = x.TokenizeSentence(d,true)


	}

	x.SaveTokens()


	data, er := json.Marshal(ds)
	check(er)
	err := os.WriteFile(output, data,0644)	
	check(err)

}


func PrepareModel(x *Tokens){
	CreateOutput(LoadDataInput("input.json"),x,"output.json")

}


const MinimumProbability float64 = 0.2
const MessageNoValidTokens string = "Sorry I am unable to help with this prompt"
const NoValidAnswers string = "All answers are lower than minimum probability"



type Model struct{
	ModelData DataSet
	Tokenizer *Tokens
}


type Answer struct {
    Text     string
    Probability float64
}

func LoadModel(file string,tokens *Tokens ) Model{
	dat, err := os.ReadFile(file)
	check(err)
	var data DataSet
    er := json.Unmarshal(dat, &data)
	check(er)
	model := Model{
		ModelData: data,
		Tokenizer:tokens,
	}
	return model

}
func findPercentageOverlap(list1, list2 []int) float64 {
    isPresent := make(map[int]bool)
    for _, item := range list2 {
        isPresent[item] = true
    }

    count := 0
    for _, item := range list1 {
        if isPresent[item] {
            count++
        }
    }

    percentage := float64(count) / float64(len(list1))
    return percentage
}


func (m *Model) Predict(input string) string{
	tokens := m.Tokenizer.TokenizeSentence(input)

	clear :=false
	for _,tok := range tokens{
		if tok != -1{
			clear = true
		}
	}
	if !clear{
		return MessageNoValidTokens
	}


	possibleAnswers := []Answer{}
	for ans, key := range m.ModelData.Data {
		percentage := findPercentageOverlap(tokens,key)
		if percentage < MinimumProbability {
			continue
		}
		possibleAnswers = append(possibleAnswers,Answer{ans,percentage})
	}

	if len(possibleAnswers) ==0 {
		return NoValidAnswers
	}

	bestAnswer := possibleAnswers[0]
	for _, answer := range possibleAnswers[1:] {
		if answer.Probability > bestAnswer.Probability {
			bestAnswer = answer
		}
	}
	// fmt.Println(bestAnswer)
	return bestAnswer.Text


}


func getInput(prompt string) string {
    reader := bufio.NewReader(os.Stdin)
    fmt.Print(prompt)

    input,err := reader.ReadString('\n')
    if err != nil {
        fmt.Println("Error reading input:", err)
        return ""
    }

    // Remove trailing newline character
    input = strings.TrimSuffix(input, "\n")

    return input
}

func staggeredPrint(x string){
	for _,i := range x{
		fmt.Print(string(i))
		time.Sleep(75 * time.Millisecond) 


	}
	fmt.Println("")
}


func main() {
	tokens := LoadTokens("tokens.json")
	PrepareModel(&tokens)



	model := LoadModel("output.json",&tokens)
	

	var inp string
	inp = getInput("Prompt: ")

	staggeredPrint(model.Predict(inp))


}	