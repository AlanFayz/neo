package main

import (
	"bufio"
	"fmt"
	"os"
	"sort"
	"strings"

	internalmath "github.com/DevAlgos/neo/source/math"
)

const (
	testInput  = "../resource/language training/book of wisdom.txt"
	characters = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n"
)

type Biagram struct {
	First, Second byte
}

type KeyValuePair struct {
	Key   Biagram
	Value int
}

func main() {
	charToToken := make(map[byte]int, 0)

	for i, char := range characters {
		charToToken[byte(char)] = i
	}

	tokenToChar := make(map[int]byte, 0)

	for r, i := range charToToken {
		tokenToChar[i] = r
	}

	bytes, err := os.ReadFile("../resource/language training/book of wisdom.txt")

	if err != nil {
		fmt.Println(err.Error())
		return
	}

	text := string(bytes)

	biagramMap := map[Biagram]int{}

	for i, _ := range text {
		first := i
		second := i + 1

		if second >= len(text) {
			break
		}

		biagram := Biagram{}
		biagram.First = text[first]
		biagram.Second = text[second]

		biagramMap[biagram] += 1
	}

	var biagramArray []KeyValuePair

	for key, val := range biagramMap {
		biagramArray = append(biagramArray, KeyValuePair{Key: key, Value: val})
	}

	sort.Slice(biagramArray, func(i, j int) bool {
		return biagramArray[i].Value > biagramArray[j].Value
	})

	matrix := internalmath.CreateMatrix(len(characters), len(characters))

	for _, biagramAndCount := range biagramArray {
		matrix.Set(charToToken[biagramAndCount.Key.First], charToToken[biagramAndCount.Key.Second], float64(biagramAndCount.Value))
	}

	const epsilon float64 = 1e-10

	for y := 0; y < len(characters); y++ {
		sum := 0.0

		for x := 0; x < len(characters); x++ {
			sum += matrix.Get(x, y)
		}

		sum = max(sum, epsilon)

		for x := 0; x < len(characters); x++ {
			current := matrix.Get(x, y)
			matrix.Set(x, y, current/sum)
		}
	}

	var n int
	reader := bufio.NewReader(os.Stdin)

	fmt.Println("Enter number of iterations: ")
	fmt.Scanf("%d\n", &n)

	fmt.Println("Enter Input: ")
	input, _ := reader.ReadString('\n')
	input = strings.TrimSpace(input)

	output := input

	fmt.Println(output)

	for i := 0; i < n; i++ {
		last := len(output) - 1

		token, exists := charToToken[byte(output[last])]
		if !exists {
			fmt.Println("Character not found in token map.")
			return
		}

		row := matrix.GetRow(token)
		index := internalmath.IndexChoice(row, 1)

		char, exists := tokenToChar[index[0]]
		if !exists {
			fmt.Println("Index out of bounds for character.")
			return
		}
		output = output + string(char)
		fmt.Println(output)
	}

}
