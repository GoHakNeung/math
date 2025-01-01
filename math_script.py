import openai
import runpy
import os
from IPython.display import display, HTML, Markdown
from google.colab import _frontend, output
from PIL import Image
import os
import pandas as pd
import shutil
from google.colab import output
import random

question_id = 0
global current_question_id
global step_content
global feedback_try
global df
global can_request_feedback
global toggle
global fourth_streak, fifth_streak, solved_correctly
global init_difficulty

output.clear()
output.no_vertical_scroll()

init_difficulty = None

fourth_streak = None
fifth_streak = None
solved_correctly = False


toggle = True
feedback_try = 0
step_content = []
can_request_feedback = False


current_question_id = question_id
df = pd.read_csv('/content/math/question_info.csv')

html_code_obj = """
<div style="display: flex; align-items: start;">
    <canvas id="canvas" style="border: 1px solid black;"></canvas>
    <div style="margin-left: 10px; display: flex; flex-direction: column;">
        <button id="undo">Undo</button>
        <button id="redo" style="margin-top: 5px;">Redo</button>
        <button id="reset" style="margin-top: 5px;">Reset</button>
    </div>
</div>

<form id="quizForm" class = "quizClass">
    <p style="text-align: left; width: 100%;">다음 중 정답을 고르세요:</p>
    <div class="labelContainer">
        <label><input type="checkbox" name="answer" value="1"> ①</label>
        <label><input type="checkbox" name="answer" value="2"> ②</label>
        <label><input type="checkbox" name="answer" value="3"> ③</label>
        <label><input type="checkbox" name="answer" value="4"> ④</label>
        <label><input type="checkbox" name="answer" value="5"> ⑤</label><br>
    </div>
</form>

<div id="assessment" class = "center">
    <button type="button" onclick="submitAnswer()">정답 확인</button>
    <div style = "width : 20px"></div>
    <button type="button" onclick="solution_feedback()">도와주세요</button>
    <div style = "width : 20px"></div>
    <button type="button" onclick="recommendQuestion()">문제 추천</button>
    <div style = "width : 20px"></div>

</div>

<style>
    .center {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 10px;
        width : 680px;
        border: 1px solid black;
        padding : 10px
    }
    .quizClass {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 10px;
        width : 680px;
        border: 1px solid black;
        padding : 10px;
        flex-direction: column
    }
    .labelContainer {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 15px;
        width: 100%;
        font-size: 20px;
    }
</style>
"""

html_code_sub = """
<div style="display: flex; align-items: start;">
    <canvas id="canvas" style="border: 1px solid black;"></canvas>
    <div style="margin-left: 10px; display: flex; flex-direction: column;">
        <button id="undo">Undo</button>
        <button id="redo" style="margin-top: 5px;">Redo</button>
        <button id="reset" style="margin-top: 5px;">Reset</button>
    </div>
</div>

<form id="quizForm" class= "quizClass">
    답 :
    <input type="text" name="answer" placeholder="답을 적으세요">
</form>

<div id="assessment" class = "center">
    <button type="button" onclick="submitAnswer()">정답 확인</button>
    <div style = "width : 20px"></div>
    <button type="button" onclick="solution_feedback()">도와주세요</button>
    <div style = "width : 20px"></div>
    <button type="button" onclick="recommendQuestion()">문제 추천</button>
    <div style = "width : 20px"></div>
</div>

<style>
    .center {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 10px;
        width : 680px;
        border: 1px solid black;
        padding : 10px
    }
    .quizClass {
        margin-top: 10px;
        width : 680px;
        border: 1px solid black;
        padding : 10px
    }
</style>
"""


def generate_html_script(question_id):
    return f"""
    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        let isDrawing = false;
        let actions = [];
        let undoneActions = [];
        let currentAction = [];
        let frameCount = 0;

        function formatFrameCount(frameNumber) {{
            return frameNumber.toString().padStart(3, '0');
        }}

        canvas.width = 700;
        canvas.height = 800;
        const folderPath = '/content/{question_id}/';

        const backgroundImage = new Image();
        backgroundImage.crossOrigin = "anonymous";
        backgroundImage.onload = function() {{
            ctx.drawImage(backgroundImage, canvas.width/2-backgroundImage.width/2, 0, backgroundImage.width, backgroundImage.height);
            saveImage('base');
        }};
        backgroundImage.src = 'https://gohakneung.github.io/math.github.io/img/question/question_{question_id}.png';

        canvas.addEventListener('pointerdown', function(e) {{
            if (e.pointerType === 'pen' || e.pointerType === 'touch' || e.pointerType === 'mouse') {{
                isDrawing = true;
                ctx.beginPath();
                ctx.moveTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
                currentAction.push({{ x: e.clientX - canvas.offsetLeft, y: e.clientY - canvas.offsetTop, type: 'start' }});

                canvas.addEventListener('pointermove', onPaint, false);
            }}
        }});

        canvas.addEventListener('pointerup', function(e) {{
            if (e.pointerType === 'pen' || e.pointerType === 'touch' || e.pointerType === 'mouse') {{
                isDrawing = false;
                actions.push([...currentAction]);
                saveStroke();
                currentAction = [];
                undoneActions = [];


                canvas.removeEventListener('pointermove', onPaint, false);
            }}
        }});

        var onPaint = function(e) {{
            if (isDrawing) {{
                const x = e.clientX - canvas.offsetLeft;
                const y = e.clientY - canvas.offsetTop;
                ctx.lineTo(x, y);
                ctx.stroke();
                currentAction.push({{ x, y, type: 'draw' }});
            }}
        }};

        function saveStroke() {{
            const filename = `image${{formatFrameCount(frameCount++)}}.png`;
            canvas.toBlob(function(blob) {{
                const reader = new FileReader();
                reader.onload = function() {{
                    const buffer = new Uint8Array(reader.result);
                    google.colab.kernel.invokeFunction('notebook.save_image', [folderPath + filename, Array.from(buffer)], {{}});
                }};
                reader.readAsArrayBuffer(blob);
            }});
        }}

        function undoLastAction() {{
            if (actions.length > 0) {{
                const lastAction = actions.pop();
                undoneActions.push(lastAction);
                redrawCanvas();
            }}
        }}

        function redoLastAction() {{
            if (undoneActions.length > 0) {{
                const actionToRedo = undoneActions.pop();
                actions.push(actionToRedo);
                redrawCanvas();
            }}
        }}

        function resetCanvas() {{
            actions = [];
            undoneActions = [];
            currentAction = [];
            redrawCanvas();
        }}

        function redrawCanvas() {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            google.colab.kernel.invokeFunction('notebook.reset_image', [], {{}});
            ctx.drawImage(backgroundImage, canvas.width/2 - backgroundImage.width/2, 0, backgroundImage.width, backgroundImage.height);
            actions.forEach(action => {{
                ctx.beginPath();
                action.forEach(point => {{
                    if (point.type === 'start') {{
                        ctx.moveTo(point.x, point.y);
                    }} else if (point.type === 'draw') {{
                        ctx.lineTo(point.x, point.y);
                        ctx.stroke();
                    }}
                }});
            }});
        }}

        function solution_feedback() {{
            google.colab.kernel.invokeFunction('notebook.output_feedback', [], {{}});
        }}

        function submitAnswer() {{
            const form = document.getElementById('quizForm');
            const formData = new FormData(form);
            const answers = formData.getAll('answer');
            const answerText = answers.join(',');
            google.colab.kernel.invokeFunction('notebook.print_to_text', [answerText], {{}});
            google.colab.kernel.invokeFunction('notebook.evaluate_answer', [], {{}});
        }}

        function saveImage(filename) {{
            canvas.toBlob(function(blob) {{
                const reader = new FileReader();
                reader.onload = function() {{
                    const buffer = new Uint8Array(reader.result);
                    google.colab.kernel.invokeFunction('notebook.save_image', ['/content/' + filename + '.png', Array.from(buffer)], {{}});
                }};
                reader.readAsArrayBuffer(blob);
            }});
        }}


        function recommendQuestion() {{
            google.colab.kernel.invokeFunction('notebook.create_scratch_cell', [], {{}});
        }}

        document.getElementById('undo').addEventListener('click', undoLastAction);
        document.getElementById('redo').addEventListener('click', redoLastAction);
        document.getElementById('reset').addEventListener('click', resetCanvas);
    </script>
    <script>
        google.colab.output.setIframeHeight(0, true);
    </script>
    """

def make_white_image() :
  img = Image.new('RGB', (700, 800), color = 'white')

  # 이미지 저장 (경로 지정 필요)
  img.save('/content/white_image.jpg')

def combine_images(base_image_path, overlay_image_path, output_image_path):
    try:
        # Open the images
        base_image = Image.open(base_image_path)
        overlay_image = Image.open(overlay_image_path)

        # Resize the overlay image to match the base image if necessary
        overlay_image = overlay_image.resize(base_image.size)

        # Paste the overlay image onto the base image
        base_image.paste(overlay_image, (0, 0), overlay_image)  # Use overlay_image as mask

        # Save the combined image
        base_image.save(output_image_path)
        print(f"Images combined and saved to {output_image_path}")

    except :
        print('파일이 없습니다.')



def save_image(path, buffer):
    buffer = bytearray(buffer)
    os.makedirs(os.path.dirname(path), exist_ok=True)  # Ensure folder creation
    with open(path, 'wb') as f:
        f.write(buffer)

def create_gif_with_pillow(duration=500):
    global current_question_id
    image_folder = f'/content/{current_question_id}'
    base_output_path = f'/content/{current_question_id}/{current_question_id}.gif'

    # 중복 파일명을 피하기 위한 함수
    def get_unique_filename(base_path):
        if not os.path.exists(base_path):
            return base_path
        index = 1
        while True:
            new_path = base_path.replace('.gif', f'_{index}.gif')
            if not os.path.exists(new_path):
                return new_path
            index += 1

    # 파일명 확인 및 유니크한 파일명 생성
    output_path = get_unique_filename(base_output_path)

    # 이미지 리스트 생성
    images = []
    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            file_path = os.path.join(image_folder, filename)
            images.append(Image.open(file_path))
    # print(images)

    # GIF 생성
    if images:
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0
        )
# Register the save_image function
output.register_callback('notebook.save_image', save_image)

# Function to display the problem dynamically
def display_problem(question_id):
    output.no_vertical_scroll()
    global fourth_streak, fifth_streak, solved_correctly
    solved_correctly = False

    global init_difficulty
    global df
    global current_question_id
    current_question_id = question_id

    if init_difficulty == None :
        init_difficulty = df[df['id'] == question_id]['difficulty'].iloc[0]

    if fourth_streak is None:
        fourth_streak = 0
    if fifth_streak is None:
        if init_difficulty ==1 :
            fifth_streak = 0
        elif init_difficulty == 2 :
            fifth_streak = 2
        elif init_difficulty == 3 :
            fifth_streak = 4

    global feedback_try
    feedback_try = 0
    global can_request_feedback
    can_request_feedback = False
    global toggle
    toggle = True
    can_request_feedback = True



    current_question_id = question_id  # Store the current question ID

    folder_path = f'/content/{question_id}'
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else :
        os.makedirs(folder_path, exist_ok=True)  # Ensure folder creation




    html_code_script = generate_html_script(question_id)

    # DataFrame에서 id 값 검색 및 type에 따른 is_objective 설정
    row = df.loc[df['id'] == current_question_id]
    if not row.empty:
        question_type = row.iloc[0]['type']

        if question_type == 'sub':
            is_objective = False
        elif question_type == 'obj':
            is_objective = True
        else:
            raise ValueError(f"Unknown type: {question_type}")
    else:
        raise ValueError(f"ID {current_question_id} not found in DataFrame")

    # Load the appropriate HTML template
    if is_objective:
        display(HTML(html_code_obj + html_code_script))
    else:
        display(HTML(html_code_sub + html_code_script))
    feedback_func()
    make_white_image()
    os.rename('/content/white_image.jpg', '/content/a.jpg')
    base_image_path = "/content/a.jpg" # base.png 파일 경로
    shutil.copy(base_image_path, folder_path)  # base.png 파일 이동

def print_to_text(text) :
    global current_question_id
    with open(f'/content/answer_{current_question_id}.txt', 'w') as f:
        f.write(text)


def evaluate_answer():
    global can_request_feedback
    global toggle
    global fourth_streak, fifth_streak, solved_correctly
    try:
        global current_question_id
        global df
        if current_question_id is None:
            print("Error: No question ID is set.")
            return

        # Read the student's answer from answer.txt
        answer_file_path = f'/content/answer_{current_question_id}.txt' #
        with open(answer_file_path, 'r') as file:
            student_answers = file.read().strip().split(',')
        # print(student_answers)
        # Find the corresponding row in the DataFrame
        question_row = df[df['id'] == current_question_id]
        if question_row.empty:
            print(f"Problem ID {current_question_id} not found in the database.")
            return
        # print(question_row)
        # Extract the correct answer(s)
        correct_answers = str(question_row.iloc[0]['answer']).split(',')
        correct_answers = [answer.strip() for answer in correct_answers]
        # print(correct_answers)
        create_gif_with_pillow()

        # Compare student's answer with the correct answer(s)
        if all(answer.strip() in correct_answers for answer in student_answers):
            print(f"정답입니다.")
            solved_correctly = True
            fifth_streak += 1
            if fifth_streak == 6 :
                fourth_streak += 1
        else:
            print(f"오답입니다.")
            solved_correctly = False
            fifth_streak -= 1
            if fifth_streak < 0 :
                fifth_streak = 0

        toggle = True
    except FileNotFoundError:
        print("Error: answer.txt file not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def feedback_func() :
    global current_question_id
    global step_content
    global df

    question_1 = df[df['id'] == current_question_id]['question']
    answer_1 = df[df['id'] == current_question_id]['solution']

    # 사용 모델을 설정합니다. chat GPT는 gpt-4o-mini를 사용합니다.
    MODEL = "gpt-4o-mini"
    USER_INPUT_MSG = f'문제는 {question_1}이고 이에 대한 해설은 {answer_1}이야. 해설을 바탕으로 문제를 해결하기 위한 피드백을 system prompt를 바탕으로 피드백을 4단계로 만들어서 줘, 피드백은 ;로 구분해서 제공해 줘'
    runpy.run_path('/content/math/mango.py')
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "너는 초등학생에 문제해결을 위한 피드백을 주는 역할을 한다. 피드백은 4단계로 제공된다. 1단계는 학생들이 문제를 이해할 수 있도록 도와준다. 2단계는 문제와 관련된 개념을 설명해 준다. 3단계는 문제 해결에 필요한 식 또는 풀이 방법을 제공해준다. 4단계는 3단계에서 제공한 식 또는 풀이 방법에 문제에서 제시된 값을 넣어서 학생들이 풀 수 있도록 한다. 구체적인 답을 주면 안되고 풀수 있도록 도와줘야 한다. 형식은 1~4단계를 쪼개야 하므로, ;로 구분지어 줘. 피드백은 최대한 간결하게 해 줘 "},
            {"role": "user", "content": USER_INPUT_MSG}
        ],
        temperature=0.2,
    )
    # print(response['choices'][0]['message']['content'])
    feedback = response['choices'][0]['message']['content'].split(';')
    # step_content = [feedback[i].strip() for i in range(len(feedback)) if feedback[i].strip().startswith(f'{i+1}단계:')] # 이렇게 한줄로 처리할 수도 있다.

    step_content = []
    print(feedback)
    print(openai.api_key)
    for i in range(len(feedback)) :
      cleaned_string = feedback[i].strip()  # 앞뒤 공백 제거
      if cleaned_string.startswith(f'{i+1}단계:'):
        step_content.append(cleaned_string)
      else:
        print(f"{i+1}단계 내용을 찾을 수 없습니다.")
    del openai.api_key


def output_feedback() :
    global step_content
    global feedback_try
    global can_request_feedback
    global toggle
    if feedback_try < 4 :
        if toggle :
            display(Markdown(step_content[feedback_try]))
            toggle = False
            feedback_try += 1
        else :
            print("정답 확인 후에, 도움을 요청하세요")
    else :
        print("모든 피드백이 이미 제공되었습니다.")

def reset_image() :
    make_white_image()

    global current_question_id
    folder_path = f'/content/{current_question_id}'
    destination_file_path = os.path.join(folder_path, 'base.jpg')

    base_image_path = '/content/white_image.jpg'
    overlay_image_path = '/content/base.png'
    combine_images(base_image_path, overlay_image_path, destination_file_path)

    # base 파일 복사

    # shutil.copy('/content/base.png', destination_file_path)

    # 파일명 변경
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith((".png", ".jpg"))]
    print("정렬전, ", files)
    # 파일 이름 정렬 (숫자가 포함된 경우에도 올바르게 정렬되도록)
    files.sort()
    print("정렬 후, ", files)
    # 2-2. 가장 마지막 .png 파일 이름에 _1 추가
    if files:
        last_file = files[-1]
        name, ext = os.path.splitext(last_file)
        new_file_name = f"{name}_1.jpg"
        new_file_path = os.path.join(folder_path, new_file_name)
        os.rename(os.path.join(folder_path, 'base.jpg'), new_file_path)


    print(f"base.png has been renamed to {new_file_name} in {folder_path}")


# 아래는 문제 추천
# Initialize streaks


def create_scratch_cell():
    global current_question_id
    global fourth_streak, fifth_streak, solved_correctly

    recommend_id = recommend_problem(current_question_id, solved_correctly, fourth_streak, fifth_streak)
    _frontend.create_scratch_cell(f"#이 코드셀을 실행하세요. \ndisplay_problem({recommend_id})")
    # next_question = recommend_next_question(globals_variable.question_num, globals_variable.final_result, question_info, attempts)
    # if next_question == '추천 문제가 없습니다.' :
    #     _frontend.create_scratch_cell(f"#{next_question}.\n#노트북으로 돌아가세요.")
    # else :
    #     _frontend.create_scratch_cell(f"#이 코드를 실행해주세요.\nQuestion('{next_question}')")



# Recommendation logic
def recommend_problem(current_problem_id, _solved_correctly, _fourth_streak, _fifth_streak):
    global df
    global fourth_streak, fifth_streak, solved_correctly
    # Get the current problem details
    current_problem = df[df['id'] == current_problem_id]

    # Check if current_problem is empty before accessing iloc[0]
    if current_problem.empty:
        print(f"Problem ID {current_problem_id} not found in the database.")
        return None  # Or handle the case appropriately

    current_problem = current_problem.iloc[0]  # Now access iloc[0] safely

    grade, area, subarea1, subarea2, content, difficulty = (
        current_problem['1st_grade'],
        current_problem['2nd_area'],
        current_problem['3rd_subarea1'],
        current_problem['4th_subarea2'],
        current_problem['5th_content'],
        current_problem['difficulty']
    )

    if solved_correctly:
        if fourth_streak >= 2:
            fourth_streak = 0
            fifth_streak = 2
            recommendations = df[
                (df['1st_grade'] == grade) &
                (df['2nd_area'] == area) &
                (df['3rd_subarea1'] == subarea1) &
                (df['4th_subarea2'] != subarea2) &
                (df['difficulty'] == 2)
           ]

        else :
            if fifth_streak == 2 :
                recommendations = df[
                    (df['1st_grade'] == grade) &
                    (df['2nd_area'] == area) &
                    (df['3rd_subarea1'] == subarea1) &
                    (df['4th_subarea2'] == subarea2) &
                    (df['5th_content'] == content) &
                    (df['difficulty'] == difficulty + 1)
                ]

            elif fifth_streak == 4 :
                recommendations = df[
                    (df['1st_grade'] == grade) &
                    (df['2nd_area'] == area) &
                    (df['3rd_subarea1'] == subarea1) &
                    (df['4th_subarea2'] == subarea2) &
                    (df['5th_content'] == content) &
                    (df['difficulty'] == difficulty + 1)
                ]

            elif fifth_streak >= 6 :
                recommendations = df[
                    (df['1st_grade'] == grade) &
                    (df['2nd_area'] == area) &
                    (df['3rd_subarea1'] == subarea1) &
                    (df['4th_subarea2'] == subarea2) &
                    (df['5th_content'] != content) &
                    (df['difficulty'] == 2)
                ]
                fifth_streak = 2


            else :
                recommendations = df[
                    (df['1st_grade'] == grade) &
                    (df['2nd_area'] == area) &
                    (df['3rd_subarea1'] == subarea1) &
                    (df['4th_subarea2'] == subarea2) &
                    (df['5th_content'] == content) &
                    (df['difficulty'] == difficulty)
                ]
        # 오답일 때 문제 추천 만들기
    else :
        if fifth_streak >= 4 :
            recommendations = df[
                (df['1st_grade'] == grade) &
                (df['2nd_area'] == area) &
                (df['3rd_subarea1'] == subarea1) &
                (df['4th_subarea2'] == subarea2) &
                (df['5th_content'] == content) &
                (df['difficulty'] == 3)
            ]

        elif fifth_streak >= 2 :
            recommendations = df[
                (df['1st_grade'] == grade) &
                (df['2nd_area'] == area) &
                (df['3rd_subarea1'] == subarea1) &
                (df['4th_subarea2'] == subarea2) &
                (df['5th_content'] == content) &
                (df['difficulty'] == 2)
            ]
        else :
            recommendations = df[
                (df['1st_grade'] == grade) &
                (df['2nd_area'] == area) &
                (df['3rd_subarea1'] == subarea1) &
                (df['4th_subarea2'] == subarea2) &
                (df['5th_content'] == content) &
                (df['difficulty'] == 1)
            ]


    print("fourth_streak, fifth_streak, solved_correctly", fourth_streak, fifth_streak, solved_correctly)
    print(recommendations)
    # Select a random problem
    if not recommendations.empty:
        recommended_problem = recommendations.sample(n=min(3, len(recommendations))).sample(1).iloc[0]
        return recommended_problem['id']
    else:
        recommendations = df[
            (df['1st_grade'] == grade) &
            (df['2nd_area'] == area) &
            (df['3rd_subarea1'] == subarea1) &
            (df['4th_subarea2'] == subarea2) &
            (df['difficulty'] == difficulty)
        ]
        recommended_problem = recommendations.sample(n=min(3, len(recommendations))).sample(1).iloc[0]
        return recommended_problem['id']





output.register_callback('notebook.create_scratch_cell', create_scratch_cell)
output.register_callback('notebook.print_to_text', print_to_text)
output.register_callback('notebook.evaluate_answer', evaluate_answer)
output.register_callback('notebook.output_feedback',output_feedback)
output.register_callback('notebook.reset_image',reset_image)
output.register_callback('notebook.recommend_problem',recommend_problem)
# Example usage
# display_problem(34301, is_objective=True)
