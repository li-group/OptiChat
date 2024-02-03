import os
import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QScrollArea, QComboBox,
                               QWidget, QTextEdit, QPushButton, QLineEdit, QFileDialog, QLabel, QGroupBox)
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QIcon, QKeySequence, QShortcut
from PySide6.QtGui import QTextCursor, QColor, QBrush, QPalette, QFont

from Util import load_model, extract_component, add_eg, read_iis
from Util import infer_infeasibility, param_in_const, extract_summary, evaluate_gpt_response, classify_question, get_completion_detailed, convert_to_standard_form, get_constraints_n_parameters, get_completion_for_quantity_sensitivity, get_variables_n_indices
from Util import get_completion_from_messages_withfn, gpt_function_call, get_parameters_n_indices, get_completion_for_index, get_completion_for_index_sensitivity, get_constraints_n_indices, get_completion_general, get_completion_from_messages_withfn_its, get_completion_for_index_variables

from enum import Enum

class Question_Type(Enum):
    ITS = "1"
    SEN = "2"
    GEN = "3"
    DET = "4"
    OPT = "5"
    OTH = "6"

class InvalidGPTResponse(Exception):
    pass

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
        self.param_names_signal.emit(param_list + param_list)
        summary = extract_summary(var_list, param_list, const_list, PYOMO_CODE, self.gpt_model)
        self.table_signal.emit(summary)
        summary_response = add_eg(summary, self.gpt_model)
        self.summary_signal.emit(summary_response)

        const_names, param_names, iis_dict = read_iis(self.ilp_path, self.model)
        iis_relation = param_in_const(iis_dict)
        self.iis_relation_signal.emit(iis_relation)
        infeasibility_report = infer_infeasibility(const_names, param_names, summary, self.gpt_model, self.model)
        self.infeasibility_report_signal.emit(infeasibility_report)


class ChatThread(QThread):
    ai_message_signal = Signal(str)
    fn_message_signal = Signal(str)
    fn_name_signal = Signal(str)

    def __init__(self, chatbot_messages, param_names_aval, model, gpt_model, py_path):
        super().__init__()
        self.chatbot_messages = chatbot_messages
        self.param_names_aval = param_names_aval
        self.model = model
        self.gpt_model = gpt_model
        self.py_path = py_path
        _, _, _, self.PYOMO_CODE = extract_component(self.model, self.py_path)
        self.model_info = get_parameters_n_indices(self.model)
        self.model_constraint_info = get_constraints_n_indices(self.model)
        self.model_constraint_parameters_info = get_constraints_n_parameters(self.model)
        self.model_variables_info = get_variables_n_indices(self.model)

        self.ai_message = None

    def run(self):
        # import pdb
        # pdb.set_trace()
        classification = classify_question(self.chatbot_messages[-1], self.gpt_model)
        print("classification", classification)

        if classification == Question_Type.ITS.value:
            response = get_completion_from_messages_withfn_its(self.chatbot_messages, self.gpt_model)
            try:
                fn_call = response.choices[0].message.tool_calls[0]
                fn_name = fn_call.function.name
                arguments = fn_call.function.arguments
                param_names = eval(arguments).get("param_names")
                for param_name in param_names:
                    if eval(f"self.model.{param_name}.is_indexed()"):
                        new_response = get_completion_for_index(self.chatbot_messages[-1], self.model_info, self.PYOMO_CODE, self.gpt_model)
                        break
                    else:
                        new_response = response
                (fn_message, flag), fn_name = gpt_function_call(new_response, self.param_names_aval, self.model)
                orig_message = {'role': 'function', 'name': fn_name, 'content': fn_message}
                self.chatbot_messages.append(orig_message)
                if flag == 'feasible':
                    expl_message = {'role': 'system',
                                    'content': 'Tell the user that you made some changed to the code and ran it, and '
                                            'the model becomes feasible. '
                                            'Replace the parameter symbol in the text with its physical meaning and mention the amount by which you changed it (if applicable)'
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
                response = get_completion_general(self.chatbot_messages, self.gpt_model)
                self.ai_message = response
                self.ai_message_signal.emit(self.ai_message)
            except:
                new_response = response.choices[0].message.content
                self.ai_message = new_response
                self.chatbot_messages.append({'role': 'assistant', 'content': new_response})
                self.ai_message_signal.emit(self.ai_message)
        elif classification == Question_Type.SEN.value:
            
            new_response = get_completion_for_index_sensitivity(self.chatbot_messages[-1], self.model_info, self.model_constraint_parameters_info, self.PYOMO_CODE, self.gpt_model)
            (fn_message, flag), fn_name = gpt_function_call(new_response, self.param_names_aval, self.model, nature='sensitivity_analysis', user_query=self.chatbot_messages[-1], gpt_model=self.gpt_model)
            orig_message = {'role': 'function', 'name': fn_name, 'content': fn_message}
            self.chatbot_messages.append(orig_message)
            if flag == 'feasible':
                #  TODO: Done
                expl_message = {'role': 'system',
                                'content': 'Tell the user that you did sensitivity analysis for the parameter they asked'
                                           'Replace the parameter symbol in the text with its physical meaning '
                                           '(for example, you could say "changing the number of men in each port" '
                                           'instead of saying "increasing demand_rule") '
                                           'and provide brief explanation.'}
            elif flag == 'infeasible':
                expl_message = {'role': 'system',
                                'content': 'Tell the user that you cannot answer such question because the model is still infeasible.'
                                           'Explain why it is not possible to perform sensitivity analysis for an infeasible model and '
                                           'suggest other ways that the user can try.'}
            elif flag == 'invalid':
                expl_message = {'role': 'system',
                                'content': 'Tell the user that you cannot change the things they requested. '
                                           'Explain why users instruction is invalid and '
                                           'suggest the parameters that the user can try.'}
            self.chatbot_messages.append(expl_message)
            response = get_completion_general(self.chatbot_messages, self.gpt_model)
            self.ai_message = response
            self.ai_message_signal.emit(self.ai_message)
        elif classification == Question_Type.GEN.value:
            print("General")
            print(self.chatbot_messages)
            print(type(self.chatbot_messages))
            new_response = get_completion_general(self.chatbot_messages, self.gpt_model)
            self.ai_message = new_response
            self.ai_message_signal.emit(self.ai_message)
        elif classification == Question_Type.DET.value:
            new_response = get_completion_detailed(self.chatbot_messages[-1], self.model_info, self.PYOMO_CODE, self.gpt_model)
            self.ai_message = new_response.choices[0].message.content
            self.ai_message_signal.emit(self.ai_message)
        elif classification == Question_Type.OPT.value:
            new_response = get_completion_for_index_variables(self.chatbot_messages[-1], self.model_variables_info, self.PYOMO_CODE, self.gpt_model)
            (fn_message, flag), fn_name = gpt_function_call(new_response, self.param_names_aval, self.model, nature='optimal_value', user_query=self.chatbot_messages[-1], gpt_model=self.gpt_model)
            orig_message = {'role': 'function', 'name': fn_name, 'content': fn_message}
            self.chatbot_messages.append(orig_message)
            if flag == 'feasible':
                expl_message = {'role': 'system',
                                'content': 'Tell the user that you ran the model and found the optimal value for the variables and the objective function they asked'
                                           'Replace the objective name and variable names in the text with its physical meaning '
                                           '(for example, you could say "the minimum budget" '
                                           'instead of saying "obj") '
                                           'and provide brief explanation.'}
            elif flag == 'infeasible':
                expl_message = {'role': 'system',
                                'content': 'Tell the user that the model is infeasible and hence you cannot find the optimal value for the variables and the objective function they asked'
                                            'Explain why it is not possible to find the optimal value for the variables and the objective function for an infeasible model and '
                                            'suggest other ways that the user can try.'}
            elif flag == 'invalid':
                expl_message = {'role': 'system',
                                'content': 'Tell the user that you cannot answer what they requested'
                                           'Explain why users instruction is invalid and '
                                           'suggest they can ask instead.'}
            self.chatbot_messages.append(expl_message)
            response = get_completion_general(self.chatbot_messages, self.gpt_model)
            self.ai_message = response
            self.ai_message_signal.emit(self.ai_message)
        else:
            # TODO: General query answering/respond "I dont know"
            expl_message = {
                'role': 'system',
                'content': """Tell the user that you do not have the capability to answer this kind of queries yet.
                Explain it to the user that you can help with any other queries regarding the model information general,
                infeasibility troubleshooting and sensitivity analysis"""

            }
            self.chatbot_messages.append(expl_message)
            response = get_completion_general(self.chatbot_messages, self.gpt_model)
            self.ai_message = response
            self.ai_message_signal.emit(self.ai_message)

class InfeasibleModelTroubleshooter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(100, 100, 1024, 768)
        self.timer = QTimer()
        self.timer.timeout.connect(self.send_text)
        # self.timer.start(100)

        self.chatbot_messages = [{'role': 'system',
                                  'content': """You are an expert in optimization and Pyomo who helps unskilled user to 
                                  troubleshoot the infeasible optimization model and any optimization questions. \n
                                  You are encouraged to remind users that they can change the value of model parameters to 
                                  make the model become feasible, but try your best to avoid those parameters that have 
                                  product with variables. \nIf the users ask you to change a parameter that has product 
                                  with variable, DO NOT use "they are parameters that have product with variables" as 
                                  explanation. Instead, you should give the physical or business context to explain why 
                                  this parameter cannot be changed. If the users keep insisting on changing the 
                                  parameter, you can try changing them but give them a warning. \n
                                  You are not allowed to have irrelevant conversation with users."""}]
        self.init_ui()

    def export_messages(self):
        chat_his = ""
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Chat History", "", "Text Files (*.txt);;All Files (*)")
        if file_path:
            with open(file_path, 'w') as file:
                for message in self.chatbot_messages:
                    if message['role'] != "system":
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
        # import pdb
        # pdb.set_trace()
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
        new_lbl.setFont(QFont("Arial", 12))
        new_lbl.setWordWrap(True)
        new_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.mid_layout.addWidget(new_lbl)
        self.current_index = 0
        self.text_to_stream = message
        self.new_lbl = new_lbl
        # self.timer.start(0.01)
        QTimer.singleShot(0, lambda: self.scroll_area.ensureWidgetVisible(new_lbl))

    def send_text(self):
        if self.current_index < len(self.text_to_stream):
            self.new_lbl.setText(self.new_lbl.text() + self.text_to_stream[self.current_index])
            self.current_index += 1
        else:
            self.timer.stop()

    
    def enter(self):
        self.lbl_model.setText(f"GPT is answering...")
        user_message = self.txt_in.toPlainText()
        self.add_message('user', user_message)
        self.txt_in.clear()
        chatbot_messages, param_names_aval, model, gpt_model = self.chatbot_messages, self.param_names, self.model, self.combobox.currentText()
        self.chat_thread = ChatThread(chatbot_messages, param_names_aval, model, gpt_model, self.py_path)

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
        self.combobox.addItem("gpt-4-1106-preview")
        self.combobox.addItem("gpt-4")
        self.combobox.addItem("gpt-3.5-turbo")
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
    QApplication.setStyle("fusion")
    app = QApplication(sys.argv)
    window = InfeasibleModelTroubleshooter()
    window.show()
    sys.exit(app.exec())
