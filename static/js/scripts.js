

const choices = ['グー', 'チョキ', 'パー'];
let gameResults = [];
let maxCount = 20;
let opponentID = generateOpponentID();
let wins = 0;
let loses = 0;
let draws = 0;
let next_choice = 0;
var socket = io();

const handImages = {
    "グー": "static/img/janken_gu.png",
    "チョキ": "static/img/janken_choki.png",
    "パー": "static/img/janken_pa.png"
};

const questionImage = "static/img/mark_question.png";

resetGame();

function playGame(playerChoice) {

    let computerChoice = next_choice;
    let result = '';
    if (playerChoice === computerChoice) {
        result = '引き分け';
        draws += 1
    } else if (
        (playerChoice === 'グー' && computerChoice === 'チョキ') ||
        (playerChoice === 'チョキ' && computerChoice === 'パー') ||
        (playerChoice === 'パー' && computerChoice === 'グー')
    ) {
        result = '勝ち';
        wins += 1
    } else {
        result = '負け';
        loses += 1
    }
    
    let sum = wins + loses + draws;
    gameResults.push([opponentID, string2number(playerChoice), string2number(computerChoice), string2number(result)]);
    document.getElementById("scores").innerHTML = `
        勝ち: ${parseInt(100*parseFloat(wins)/sum)}%, 
        負け: ${parseInt(100*parseFloat(loses)/sum)}%, 
        引き分け: ${parseInt(100*parseFloat(draws)/sum)}%`;
    document.getElementById("player-hand").src = handImages[playerChoice];
    document.getElementById("computer-hand").src = handImages[computerChoice];
    document.getElementById("wld").innerHTML = result;

    if (gameResults.length === maxCount) {
        resetGame();

        alert("勝負が終了しました．リセットします．");
    }

    chooseNext(playerChoice);
    console.log(next_choice);
}

function chooseNext(playerChoice) {
    socket.emit('choose', string2number(playerChoice), 
        function(response) {
            next_choice = number2string(response);
    });
}


function generateOpponentID() {
    return Math.floor(Math.random() * 100000);
}

function resetGame() {
    gameResults = [];
    draws = 0;
    wins = 0;
    loses = 0;
    opponentID = generateOpponentID();
    document.getElementById("scores").innerHTML = `勝ち: ${wins}/${maxCount}, 負け: ${loses}/${maxCount}, 引き分け: ${draws}/${maxCount}`;
    document.getElementById("player-hand").src = questionImage;
    document.getElementById("computer-hand").src = questionImage;
    document.getElementById("wld").innerHTML = "";
    socket.emit('reset');
    socket.emit('save_data');
    next_choice = 0;
}

function string2number(str) {
    if (str === "グー") {
        return 0;
    } else if (str === "チョキ") {
        return 1;
    } else if (str === "パー") {
        return 2;
    }

    if (str === "勝ち") {
        return 1;
    } else if (str === "引き分け") {
        return 0;
    } else if (str === "負け") {
        return -1;
    }
}
function number2string(num) {
    if (num === 0) {
        return "グー";
    } else if (num === 1) {
        return "チョキ";
    } else if (num === 2) {
        return "パー";
    }
}



function sendResultsToSheet() {
    fetch("https://script.google.com/macros/s/AKfycbz1GBmKlrfGWxXjNBxE9XiRWZ6yR6ul0x92qaZJVJicBrdS_qxVRGTuIZ8gV6I7PaVL/exec", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        mode: "no-cors",
        body: JSON.stringify(gameResults)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => alert("データをスプレッドシートに送信しました！"))
    .catch(error => {
        console.error("fetch エラー:", error);
    });
}
