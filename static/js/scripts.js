

const choices = ['グー', 'チョキ', 'パー'];
let gameResults = [];
let maxCount = 10;
let opponentID = generateOpponentID();
let wins = 0;
let loses = 0;
let draws = 0;
let next_choice = "グー";
var socket = io();

const sleep = (time) => new Promise((resolve) => setTimeout(resolve, time));//timeはミリ秒


const handImages = {
    "グー": "static/img/janken_gu.png",
    "チョキ": "static/img/janken_choki.png",
    "パー": "static/img/janken_pa.png",
    "Unknown": "static/img/mark_question.png",
};

// ゲームの流れはここで実装
async function startGame() {
    changeScreen("game-screen");
    resetGame();

    for(let i=0; i<maxCount; i++){
        document.getElementById("progress-bar").style.width = ((100*i)/maxCount)+"%";
        await startGauge();
        document.getElementById("computer-hand").src = handImages[next_choice];

        socket.emit("capture_hand");
        await new Promise((resolve) => {
            socket.once('capture_done', (response)=>{
                playerChoice = response.gesture;
                if(playerChoice === "Unknown"){
                    alert("手が認識できませんでした．もう一度やり直してください．");
                    i--;
                }else{
                    playGame(playerChoice);
                    console.log("win" + wins + " lose" + loses + " draw" + draws);
                }
                resolve();
            });
        });
    }

    await updata_score(wins, draws, loses);
    await sleep(1000);
    changeScreen("result-screen");
    resetGame();
}

// じゃんけんパート
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
    
    gameResults.push([opponentID, string2number(playerChoice), string2number(computerChoice), string2number(result)]);
    document.getElementById("wld").innerHTML = result;

    console.log("cpu: " + computerChoice);
    console.log("player: " + playerChoice);
    chooseNext(playerChoice, computerChoice);
    console.log("next_choice: " + next_choice);
}

function updata_score(wins, draws, loses) {
    let sum = wins + loses + draws;
    document.getElementById("scores").innerHTML = `
        勝ち: ${parseInt(100*parseFloat(wins)/sum)}%, 
        負け: ${parseInt(100*parseFloat(loses)/sum)}%, 
        引き分け: ${parseInt(100*parseFloat(draws)/sum)}%`;
    if(wins > loses) {
        document.getElementById("result-message").innerHTML = "あなたの勝ち！";
        document.getElementById("result-message").style.color = "red";
    }
    else if(wins < loses) {
        document.getElementById("result-message").innerHTML = "あなたの負け！";
        document.getElementById("result-message").style.color = "blue";
    }
    else {
        document.getElementById("result-message").innerHTML = "引き分け！";
        document.getElementById("result-message").style.color = "black";
    }
}

function chooseNext(playerChoice, computerChoice) {
    socket.emit('choose', {var1:string2number(playerChoice), var2:string2number(computerChoice)}, 
        function(response) {
            next_choice = number2string(response);
    });
}

function generateOpponentID() {
    return Math.floor(Math.random() * 100000);
}

async function resetGame() {
    gameResults = [];
    draws = 0;
    wins = 0;
    loses = 0;
    document.getElementById("computer-hand").src = handImages["Unknown"];
    document.getElementById("wld").innerHTML = " ";
    socket.emit('save_data');
    await new Promise((resolve) => {
        socket.once('save_done', resolve);
    });
    socket.emit('reset');
    next_choice = "グー";
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

function startGauge(){
    let progress = 0;
    document.getElementById("gauge-bar").style.width = "0%";
    document.getElementById("computer-hand").src = handImages["Unknown"];
    return new Promise((resolve) => {
        let interval = setInterval(function() {
            progress += 2;
            document.getElementById("gauge-bar").style.width = progress + "%";
            if (progress > 100) {
                document.getElementById("gauge-bar").style.width = "0%";
                clearInterval(interval);
                resolve();
            }
        }, 60);
    });
}

function changeScreen(screenId){
    document.querySelectorAll('.screen').forEach(screen => {
        screen.classList.remove('active');
    });
    document.getElementById(screenId).classList.add('active');
}