import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain import LLMChain
from dotenv import load_dotenv

load_dotenv()
# -------------------------------
# LLM応答を生成する関数
# -------------------------------
def get_expert_response(user_input: str, expert_type: str) -> str:
    """入力テキストと専門家タイプを基にLLMからの応答を返す"""

    # 専門家ごとのシステムメッセージ
    if expert_type == "健康アドバイザー":
        system_template = """
        あなたは信頼できる健康アドバイザーです。
        食生活、運動、睡眠、ストレス管理について、
        科学的根拠に基づいたアドバイスを日本語で分かりやすく説明してください。
        """
    elif expert_type == "旅行プランナー":
        system_template = """
        あなたは経験豊富な旅行プランナーです。
        ユーザーの希望に合わせて、観光地の紹介、モデルコース、費用の目安などを
        わかりやすく提案してください。日本語で答えてください。
        """
    else:
        system_template = "あなたはユーザーに役立つ回答を行うアシスタントです。"

    # LLMとプロンプトを準備
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", "{input_text}")
    ])
    chain = LLMChain(prompt=prompt, llm=llm)

    # 実行
    response = chain.run(input_text=user_input)
    return response


# -------------------------------
# Streamlitアプリ本体
# -------------------------------
def main():
    st.title("LLM搭載 Webアプリ")

    st.markdown("""
    ### アプリ概要
    このアプリは、入力フォームからテキストを送信すると、選択した専門家の立場からLLMが回答を生成してくれるデモです。  
    - 下のラジオボタンで **専門家の種類** を選びます  
    - テキストを入力して送信すると、専門家の立場に応じた回答が表示されます  
    """)

    # 専門家の選択
    expert_type = st.radio(
        "専門家の種類を選択してください：",
        ("健康アドバイザー", "旅行プランナー")
    )

    # 入力フォーム
    user_input = st.text_area("質問や相談内容を入力してください：")

    if st.button("送信"):
        if user_input.strip():
            with st.spinner("LLMが考えています..."):
                response = get_expert_response(user_input, expert_type)
            st.subheader("回答結果")
            st.write(response)
        else:
            st.warning("テキストを入力してください。")


if __name__ == "__main__":
    main()