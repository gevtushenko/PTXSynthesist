#include "main_window.h"

#include <QFile>
#include <QTimer>
#include <QToolBar>
#include <QGroupBox>
#include <QLineEdit>
#include <QComboBox>
#include <QTextEdit>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QValueAxis>
#include <QHeaderView>
#include <QTableWidget>
#include <QBarCategoryAxis>

#include "ptx_executor.h"
#include "ptx_generator.h"
#include "ptx_interpreter.h"

#include "code_editor.hpp"
#include "cuda_highlighter.hpp"
#include "ptx_highlighter.hpp"
#include "syntax_style.h"

ScatterLineSeries::ScatterLineSeries()
    : line_series(new QLineSeries())
    , scatter_series(new QScatterSeries())
{ }

void ScatterLineSeries::set_color(QColor color)
{
    QPen pen = line_series->pen();
    pen.setWidth(3);
    pen.setBrush(QBrush(color));

    line_series->setPen(pen);
    scatter_series->setColor(color);
    // scatter_series->setColor(QColor("#F8F8F2"));
}

void ScatterLineSeries::add_to_chart(QChart *chart)
{
    chart->addSeries(line_series);
    chart->addSeries(scatter_series);
}

void ScatterLineSeries::append(int x, float y)
{
    line_series->append(static_cast<double>(x), y);
    scatter_series->append(static_cast<double>(x), y);
}

MainWindow::MainWindow()
  : options(new QLineEdit())
  , cuda(new CodeEditor())
  , ptx(new CodeEditor())
  , timer(new QTimer())
{
  options->setText("-lineinfo, --gpu-architecture=compute_86");

  cuda->setAcceptRichText(false);
  cuda->setPlainText("\n"
                     "// Vector add example\n"
                     "extern \"C\" __global__ void kernel(\n"
                     "  int n, \n"
                     "  const int *x, \n"
                     "  const int *y, \n"
                     "  int *result)\n"
                     "{\n"
                     "  const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;\n"
                     "  \n"
                     "  if (i < n) \n"
                     "  {\n"
                     "    result[i] = x[i] + y[i];\n"
                     "  }\n"
                     "}\n");

  setup = new QGroupBox();
  params_table = new QTableWidget(2, 4);
  params_table->verticalHeader()->setVisible(false);
  params_table->horizontalHeader()->setVisible(false);
  params_table->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
  params_table->setSelectionMode(QAbstractItemView::NoSelection);

  QVBoxLayout *group_box_layput = new QVBoxLayout();
  group_box_layput->addWidget(params_table);
  setup->setLayout(group_box_layput);

  QHBoxLayout *h_layout = new QHBoxLayout();
  h_layout->addWidget(setup, 3);
  h_layout->addWidget(cuda, 4);
  h_layout->addWidget(ptx, 4);

  QVBoxLayout *v_layout = new QVBoxLayout();
  v_layout->addWidget(options);
  v_layout->addLayout(h_layout);

  chart = new QChart();
  chart->setBackgroundBrush(QBrush(QColor("#282a36")));

  chart_view = new QChartView(chart);
  chart_view->setBackgroundBrush(QBrush(QColor("#282a36")));
  chart_view->setRenderHint(QPainter::Antialiasing);
  chart_view->hide();

  min_series.set_color("#BD93F9");
  max_series.set_color("#FFB86C");
  median_series.set_color("#50FA7B");

  min_series.add_to_chart(chart);
  max_series.add_to_chart(chart);
  median_series.add_to_chart(chart);

  chart->createDefaultAxes();
  chart->legend()->hide();

  chart->axes(Qt::Vertical).back()->setLabelsColor(QColor("#F8F8F2"));
  chart->axes(Qt::Horizontal).back()->setLabelsColor(QColor("#F8F8F2"));

  v_layout->addWidget(chart_view);

  QWidget *central_widget = new QWidget();
  central_widget->setLayout(v_layout);

  setCentralWidget(central_widget);

  timer->setSingleShot(true);

  tool_bar = addToolBar("Interpreter");
  QWidget* spacer = new QWidget();
  spacer->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  tool_bar->addWidget(spacer);

  run_action = new QAction(QIcon(":/icons/play.png"), "Run", this);
  interpret_action = new QAction(QIcon(":/icons/bug.png"), "Interpret", this);

  tool_bar->addAction(run_action);
  tool_bar->addAction(interpret_action);
  tool_bar->setMovable(false);

  QObject::connect(interpret_action, &QAction::triggered, this, &MainWindow::interpret);
  QObject::connect(run_action, &QAction::triggered, this, &MainWindow::execute);

  QObject::connect(cuda->document(), &QTextDocument::contentsChanged, this, &MainWindow::reset_timer);
  QObject::connect(options, &QLineEdit::textChanged, this, &MainWindow::reset_timer);
  QObject::connect(timer, &QTimer::timeout, this, &MainWindow::regen_ptx);

  QFont jet_brains_mono = QFont("JetBrains Mono", 12);
  cuda->setFont(jet_brains_mono);
  options->setFont(jet_brains_mono);

  cuda->setAutoIndentation(true);
  cuda->setAutoParentheses(true);
  cuda->setTabReplace(true);
  cuda->setTabReplaceSize(2);

  ptx->setFont(jet_brains_mono);
  ptx->setReadOnly(true);

  load_style(":/style/dracula.xml");

  cuda->setHighlighter(new CUDAHighlighter());
  ptx->setHighlighter(new PTXHighlighter());

  setStyleSheet("QToolBar {"
                " background-color: #414450;"
                "}"
                ""
                "QTableWidget {"
                " background-color: #414450;"
                " color: #F8F8F2; "
                "}"
                ""
                "QGroupBox {"
                " background-color: #414450;"
                " color: #F8F8F2; "
                "}"
                ""
                "QComboBox {"
                " background-color: #414450;"
                " color: #F8F8F2; "
                "}"
                ""
                "QToolButton {"
                " max-width: 15px; "
                " max-height: 15px; "
                " margin: 8px 4px; "
                "}"
                ""
                "QToolButton:hover {"
                " background-color: #282a36; "
                "}"
                ""
                "QLineEdit {"
                " background-color: #282a36; "
                " color: #F8F8F2; "
                "}"
                ""
                "QScrollBar:vertical {"
                "    padding: 0px 0px 0px 0px;"
                "    background: #282a36;"
                "}"
                "QScrollBar::handle:vertical {"
                "    background: #4c4d56;"
                "}"
                ""
                "QMainWindow {"
                " background-color: #414450; "
                "}");

  reset_timer();
}

MainWindow::~MainWindow() = default;

#include <iostream>
void MainWindow::parse_params()
{
  QString cuda_code = cuda->toPlainText();

  const int global_idx = cuda_code.indexOf("__global__");
  const int params_start = cuda_code.indexOf('(', global_idx) + 1;
  const int params_len = cuda_code.indexOf(')', params_start) - params_start;

  QString params = cuda_code.mid(params_start, params_len);
  QStringList params_list = params.split(',');

  params_list.push_front("gridDim.x");
  params_list.push_front("blockDim.x");

  std::set<QString> params_set;

  for (auto &param: params_list)
  {
    param = param.trimmed();
    params_set.insert(param);
  }

  for (const QString &param: params_set)
  {
    auto it = params_map.find(param);

    if (it == params_map.end())
    {
      params_map[param] = InputType::unspecified;
    }
  }

  params_table->setRowCount(0);

  QStringList array_input_types;
  array_input_types << "unspecified"
                    << "memset"
                    << "file";

  QStringList scalar_input_types;
  scalar_input_types << "fixed"
                     << "range";

  QAbstractItemModel *model = params_table->model();
  for (const QString &param: params_list)
  {
    const int insert_idx = params_table->rowCount();
    params_table->insertRow(insert_idx);

    model->setData(model->index(insert_idx, 0), param);
    QComboBox *input_type = new QComboBox();

    if (!param.contains('*'))
    {
      input_type->addItems(scalar_input_types);
      params_table->setSpan(insert_idx, 2, 1, 2);

      if (param == "blockDim.x")
      {
        model->setData(model->index(insert_idx, 2), "[128, 256, 512, 1024]");
      }
      else if (param == "gridDim.x")
      {
        model->setData(model->index(insert_idx, 2), "n");
      }
      else if (param.endsWith(" n"))
      {
        model->setData(model->index(insert_idx, 2), "[2 ** x for x in range(16, 27)]");
      }
    }
    else
    {
      input_type->addItems(array_input_types);
    }

    params_table->setCellWidget(insert_idx, 1, input_type);
  }
}

void MainWindow::load_style(QString path)
{
    QFile fl(path);

    if (!fl.open(QIODevice::ReadOnly))
    {
        return;
    }

    auto style = new SyntaxStyle(this);

    if (!style->load(fl.readAll()))
    {
        delete style;
        return;
    }

    cuda->setSyntaxStyle(style);
    ptx->setSyntaxStyle(style);
}

void MainWindow::interpret()
{
  PTXInterpreter interpreter;

  interpreter.interpret(ptx->toPlainText().toStdString());
}

void MainWindow::reset_timer()
{
  timer->start(1000);

  run_action->setEnabled(false);
  interpret_action->setEnabled(false);

  parse_params();
}

void MainWindow::regen_ptx()
{
  std::string cuda_source = cuda->toPlainText().toStdString();

  QStringList options_list = options->text().split(',');
  std::vector<std::string> str_options_list(options_list.size());
  std::vector<const char *> c_str_options_list(options_list.size());

  for (int i = 0; i < static_cast<int>(c_str_options_list.size()); i++)
  {
    str_options_list[i] = options_list[i].toStdString();
    c_str_options_list[i] = str_options_list[i].c_str();
  }

  PTXGenerator generator;
  std::optional<PTXCode> ptx_code = generator.gen(cuda_source.c_str(), c_str_options_list);

  if (ptx_code)
  {
    ptx->setPlainText(ptx_code->get_ptx());
  }
  else
  {
    ptx->setPlainText("Compilation error");
  }

  run_action->setEnabled(true);
  interpret_action->setEnabled(true);
}

float avg(const std::vector<float> &measurements)
{
    const float sum = std::accumulate(measurements.begin(), measurements.end(), 0.0f);
    return sum / measurements.size();
}

float median(int begin, int end, const std::vector<float> &sorted_list)
{
    int count = end - begin;

    if (count % 2)
    {
        return sorted_list.at(count / 2 + begin);
    }
    else
    {
        float right = sorted_list.at(count / 2 + begin);
        float left = sorted_list.at(count / 2 - 1 + begin);
        return (right + left) / 2.0;
    }
}

void MainWindow::execute()
{
    execution_id++;

    if (!executor)
    {
        executor = std::make_unique<PTXExecutor>();
        chart_view->show();
    }

    std::string ptx_code = ptx->toPlainText().toStdString();

    // TODO Parameter manager
    void* kernel_args[4];

    const int iterations = 10;

    std::vector<float> elapsed_times = executor->execute(
            iterations,
            kernel_args,
            256,
            256 * 1024,
            ptx_code.c_str());

    std::sort(elapsed_times.begin(), elapsed_times.end());

    const float min_time = elapsed_times.front();
    const float max_time = elapsed_times.back();

    min_elapsed = std::min(min_elapsed, min_time);
    max_elapsed = std::max(max_elapsed, max_time);

    min_series.append(execution_id, min_time);
    max_series.append(execution_id, max_time);
    median_series.append(execution_id, median(0, elapsed_times.size(), elapsed_times));

    chart->axes(Qt::Horizontal).back()->setRange(0, execution_id + 1);
    chart->axes(Qt::Vertical).back()->setRange(min_elapsed - min_elapsed / 4, max_elapsed + max_elapsed / 4);
}
