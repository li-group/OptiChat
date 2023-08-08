import os
import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QScrollArea, QComboBox,
                               QWidget, QTextEdit, QPushButton, QLineEdit, QFileDialog, QLabel, QGroupBox)
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QIcon, QKeySequence, QShortcut
from PySide6.QtGui import QTextCursor, QColor, QBrush

from Util import load_model, extract_component, add_eg, read_iis
from Util import infer_infeasibility, param_in_const, extract_summary
from Util import get_completion_from_messages_withfn, gpt_function_call


class Combobox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet('''
            QComboBox {
                font-size: 12px; 
                font-weight: bold; 
                background-color: blue
                border: 2px solid white;
                border-radius: 5px;
            }
        ''')


class BrowseButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet('''
            QPushButton {
                font-size: 12px; 
                font-weight: bold; 
                background-color: white; 
                border: 2px solid white;
                border-radius: 5px;
            }
        ''')


class OutLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet('''
            QLabel {
                border-radius: 5px; 
                padding: 10px;
            }
        ''')


class OutLabels(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet('''
            QWidget {
                background-color: white;
                border: 2px solid white;
                border-radius: 5px; 
                padding: 10px;
            }
        ''')


class OutArea(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet('''
            QScrollArea {
                background-color: white;
                border: 2px solid white;
                border-radius: 5px; 
                padding: 5px;
            }
        ''')


class InTextEdit(QTextEdit):
    enterPressed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet('''
            QTextEdit {
                min-height: 50px;
                max-height: 100px;
                font-size: 16px; 
                background-color: white; 
                border: 2px solid white;
                border-radius: 5px;
                padding: 0px;
            }
        ''')

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self.enterPressed.emit()
        else:
            super().keyPressEvent(event)


class EnterButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet('''
            QPushButton {
                background-color: white; 
                border: 2px solid white;
                border-radius: 5px;
            }
        ''')


class InGroupBox(QGroupBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet('''
            QGroupBox { 
                min-height: 55px;
                max-height: 105px;
                border-radius: 5px; 
                background-color: white; 
            }
        ''')


class ProcessThread(QThread):
    table_signal = Signal(str)
    summary_signal = Signal(str)
    infeasibility_report_signal = Signal(str)
    iis_relation_signal = Signal(str)
    param_names_signal = Signal(list)

    def __init__(self, py_path, ilp_path, model, gpt_model):
        super().__init__()
        self.py_path = py_path
        self.ilp_path = ilp_path
        self.model = model
        self.gpt_model = gpt_model

    def run(self):
        const_list, param_list, var_list, PYOMO_CODE = extract_component(self.model, self.py_path)
        self.param_names_signal.emit(param_list)
        summary = extract_summary(var_list, param_list, const_list, PYOMO_CODE, self.gpt_model)
        self.table_signal.emit(summary)
        summary_response = add_eg(summary, self.gpt_model)
        self.summary_signal.emit(summary_response)

        const_names, param_names, iis_dict = read_iis(self.ilp_path, self.model)
        iis_relation = param_in_const(iis_dict)
        self.iis_relation_signal.emit(iis_relation)
        # print(const_names, param_names, iis_dict)
        infeasibility_report = infer_infeasibility(const_names, param_names, summary, self.gpt_model)
        self.infeasibility_report_signal.emit(infeasibility_report)


class ChatThread(QThread):
    ai_message_signal = Signal(str)
    fn_message_signal = Signal(str)
    fn_name_signal = Signal(str)

    def __init__(self, chatbot_messages, param_names_aval, model, gpt_model):
        super().__init__()
        self.chatbot_messages = chatbot_messages
        self.param_names_aval = param_names_aval
        self.model = model
        self.gpt_model = gpt_model

    def run(self):
        response = get_completion_from_messages_withfn(self.chatbot_messages, self.gpt_model)
        if "function_call" not in response["choices"][0]["message"]:
            ai_message = response["choices"][0]["message"]["content"]
            self.ai_message_signal.emit(ai_message)
        else:
            (fn_message, flag), fn_name = gpt_function_call(response, self.param_names_aval, self.model)
            orig_message = {'role': 'function', 'name': fn_name, 'content': fn_message}
            self.chatbot_messages.append(orig_message)
            if flag == 'feasible':
                expl_message = {'role': 'system',
                                'content': 'Tell the user that you made some changed to the code and ran it, and '
                                           'the model becomes feasible. '
                                           'Replace the parameter symbol in the text with its physical meaning '
                                           '(for example, you could say "increasing the amount of cost invested" '
                                           'instead of saying "increasing c") '
                                           'and provide brief explanation.'}
            elif flag == 'infeasible':
                expl_message = {'role': 'system',
                                'content': 'Tell the user that you made some changed to the code and ran it, but '
                                           'the model is still infeasible. '
                                           'Explain why it does not become feasible and '
                                           'suggest other parameters that the user can try.'}
            elif flag == 'invalid':
                expl_message = {'role': 'system',
                                'content': 'Tell the user that you cannot change the things they requested. '
                                           'Explain why users instruction is invalid and '
                                           'suggest the parameters that the user can try.'}
            self.chatbot_messages.append(expl_message)
            response = get_completion_from_messages_withfn(self.chatbot_messages, self.gpt_model)
            ai_message = response["choices"][0]["message"]["content"]
            self.ai_message_signal.emit(ai_message)


class InfeasibleModelTroubleshooter(QMainWindow):
    def __init__(self):
        super().__init__()

        self.chatbot_messages = [{'role': 'system',
                                  'content': 'You are an expert in optimization who helps unskilled user to '
                                             'troubleshoot the infeasible optimization model '
                                             'and any optimization questions. '
                                             'You are encouraged to remind users that they can change the value of '
                                             'parameters to make the model become feasible. '
                                             'You are not allowed to have irrelevant conversation with users.'}]
        self.init_ui()

    def export_messages(self):
        chat_his = ""
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Chat History", "", "Text Files (*.txt);;All Files (*)")
        if file_path:
            with open(file_path, 'w') as file:
                for message in self.chatbot_messages:
                    chat_his = chat_his + '<<< ' + message['role'] + ": " + message['content'] + "\n\n\n\n"
                file.write(chat_his)
        self.lbl_model.setText(f"Chat history exported.")

    def browse_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Python Files (*.py)", options=options)

        if file_name:
            self.lbl_model.setText(f"Selected File: {file_name}")
            self.py_path = file_name

    def process_file(self):
        self.lbl_model.setText(f"Loading the model...")
        self.model, self.ilp_path = load_model(self.py_path)

        py_path, ilp_path, model, gpt_model = self.py_path, self.ilp_path, self.model, self.combobox.currentText()
        self.process_thread = ProcessThread(py_path, ilp_path, model, gpt_model)

        self.process_thread.table_signal.connect(self.process_table)
        self.process_thread.summary_signal.connect(self.process_summary)
        self.process_thread.param_names_signal.connect(self.process_names)
        self.process_thread.iis_relation_signal.connect(self.process_relation)
        self.process_thread.infeasibility_report_signal.connect(self.process_report)
        self.process_thread.finished.connect(self.process_finished)
        self.process_thread.start()

    def process_table(self, table):
        self.table = table
        self.chatbot_messages.append({'role': 'system', 'content': self.table})

    def process_summary(self, summary):
        self.summary = summary

    def process_names(self, names):
        self.param_names = names
        print(f"type of param_names: {type(self.param_names)}")
        print(f"names of param_names: {self.param_names}")

    def process_relation(self, relation):
        self.relation = relation

    def process_report(self, infeasibility_report):
        self.infeasibility_report = infeasibility_report

    def process_finished(self):
        self.add_message('assistant', self.summary + '\n' 
                         '\n' + self.infeasibility_report)
        self.lbl_model.setText(f"GPT responded.")
        self.lbl_model.setStyleSheet("color: black;")
        self.txt_in.setReadOnly(False)
        self.txt_in.setPlaceholderText("Enter")

    def add_message(self, role, message, fn_name=None):
        role_style = {
            'function': "font-size: 16px; background-color: white; border: 2px solid white",
            'assistant': "font-size: 16px; background-color: white; border: 2px solid white",
            'user': "font-size: 16px; background-color: silver; border: 2px solid silver;"
        }
        if role == 'function':
            self.chatbot_messages.append({'role': role, 'name': f"{fn_name}", 'content': f"{message}"})
        else:
            self.chatbot_messages.append({'role': role, 'content': f"{message}"})
        print(self.chatbot_messages)
        new_lbl = OutLabel(message)
        new_lbl.setStyleSheet(role_style[role])
        new_lbl.setWordWrap(True)
        new_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.mid_layout.addWidget(new_lbl)
        QTimer.singleShot(0, lambda: self.scroll_area.ensureWidgetVisible(new_lbl))

    def enter(self):
        self.lbl_model.setText(f"GPT is answering...")
        user_message = self.txt_in.toPlainText()
        self.add_message('user', user_message)
        self.txt_in.clear()
        chatbot_messages, param_names_aval, model, gpt_model = self.chatbot_messages, self.param_names, self.model, self.combobox.currentText()
        self.chat_thread = ChatThread(chatbot_messages, param_names_aval, model, gpt_model)

        self.chat_thread.ai_message_signal.connect(self.chat_ai_message)
        self.chat_thread.fn_message_signal.connect(self.chat_fn_message)
        self.chat_thread.fn_name_signal.connect(self.chat_fn_name)
        self.chat_thread.finished.connect(self.chat_finished)
        self.chat_thread.start()

    def chat_ai_message(self, ai_message):
        self.ai_message = ai_message
        self.add_message('assistant', self.ai_message)

    def chat_fn_message(self, fn_message):
        self.fn_message = fn_message

    def chat_fn_name(self, fn_name):
        self.fn_name = fn_name
        self.add_message('function', self.fn_message, self.fn_name)

    def chat_finished(self):
        self.lbl_model.setText(f"GPT responded.")

    def toggle_btn_enter(self):
        self.btn_enter.setEnabled(bool(len(self.txt_in.toPlainText()) > 0))

    def init_ui(self):
        self.setWindowTitle("Infeasible Model Troubleshooter")
        self.setGeometry(100, 100, 1200, 900)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        self.lbls_widget = OutLabels()
        self.scroll_area = OutArea()
        self.scroll_area.setWidgetResizable(True)
        group_box = InGroupBox()

        # Combobox, Browse Button, Model Label, Process Button
        self.combobox = Combobox()
        self.combobox.setFixedWidth(180)
        self.combobox.addItem("gpt-4")
        self.combobox.addItem("gpt-3.5-turbo-16k")
        self.combobox.setCurrentIndex(0)
        self.lbl_model = QLabel("Selected File: ", self)
        self.btn_browse = BrowseButton("Browse")
        self.btn_browse.setFixedSize(70, 30)
        self.btn_browse.clicked.connect(self.browse_file)
        self.btn_process = BrowseButton("Process")
        self.btn_process.setFixedSize(70, 30)
        self.btn_process.clicked.connect(self.process_file)

        # Export Button
        self.btn_export = BrowseButton("Export")
        self.btn_export.setFixedSize(70, 30)
        self.btn_export.clicked.connect(self.export_messages)

        # Input Text with rounded corners
        self.txt_in = InTextEdit()
        self.txt_in.textChanged.connect(self.toggle_btn_enter)
        self.btn_enter = EnterButton(self)
        self.btn_enter.setFixedSize(40, 40)
        # icon = QIcon(os.path.abspath('enter.png'))
        # print(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'enter.png'))
        icon = QIcon(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'enter.png'))
        self.btn_enter.setIcon(icon)
        self.btn_enter.clicked.connect(self.enter)
        self.btn_enter.setDisabled(True)
        # Enable key Enter
        self.txt_in.enterPressed.connect(self.btn_enter.click)
        self.txt_in.setReadOnly(True)
        self.txt_in.setPlaceholderText("Load your model first")

        # Layout
        # Top Layout
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.combobox)
        top_layout.addWidget(self.btn_browse)
        top_layout.addWidget(self.lbl_model)
        top_layout.addWidget(self.btn_process)
        top_layout.addWidget(self.btn_export)
        # Bot Layout
        bot_layout = QHBoxLayout(group_box)
        bot_layout.addWidget(self.txt_in)
        bot_layout.addWidget(self.btn_enter)

        # Mid Layout
        self.scroll_area.setWidget(self.lbls_widget)
        self.mid_layout = QVBoxLayout(self.lbls_widget)

        # Main Layout
        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.scroll_area)
        main_layout.addWidget(group_box)

        central_widget.setLayout(main_layout)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = InfeasibleModelTroubleshooter()
    window.show()
    sys.exit(app.exec())