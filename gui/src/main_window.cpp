#include "main_window.h"

#include <QFile>
#include <QTimer>
#include <QToolBar>
#include <QGroupBox>
#include <QLineEdit>
#include <QComboBox>
#include <QTextEdit>
#include <QValueAxis>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QHeaderView>
#include <QDockWidget>
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
  : cuda(new CodeEditor())
  , ptx(new CodeEditor())
  , timer(new QTimer())
{
  cuda->setAcceptRichText(false);
  cuda->setPlainText("\n"
                     "// Vector add example\n"
                     "\n"
                     "// threads_in_block = 256\n"
                     "// blocks_in_grid = (n + threads_in_block - 1) // threads_in_block\n"
                     "// iterations = 10\n"
                     "\n"
                     "extern \"C\" __global__ void kernel(\n"
                     "  int n        /* [2 ** x for x in range(18, 27)] */, \n"
                     "  const int *x /* n * 4   */, \n"
                     "  const int *y /* n * 4   */, \n"
                     "  int *result  /* n * 4   */)\n"
                     "{\n"
                     "  for (int i = threadIdx.x + blockIdx.x * blockDim.x;\n"
                     "       i < n;\n"
                     "       i += blockDim.x * gridDim.x) \n"
                     "  {\n"
                     "    result[i] = x[i] + y[i];\n"
                     "  }\n"
                     "}\n");

  add_editor();

  chart = new QChart();
  chart->setBackgroundBrush(QBrush(QColor("#282a36")));

  chart_view = new QChartView(chart);
  chart_view->setBackgroundBrush(QBrush(QColor("#282a36")));
  chart_view->setRenderHint(QPainter::Antialiasing);
  // chart_view->setMaximumHeight(10);
  // chart->setMaximumHeight(10);

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
  QObject::connect(timer, &QTimer::timeout, this, &MainWindow::regen_ptx);

  load_style(":/style/dracula.xml");

  cuda->setHighlighter(new CUDAHighlighter());
  ptx->setHighlighter(new PTXHighlighter());

  setStyleSheet("QToolBar {"
                " background-color: #414450;"
                "}"
                ""
                "QDockWidget {"
                " background-color: #414450;"
                " color: #414450; "
                "}"
                ""
                "QTabBar::tab {"
                " color: #F8F8F2; "
                " background: #44475a; "
                " border: 0.5px solid #f8f8f2;\n"
                " min-width: 8ex;\n"
                " padding: 2px;\n"
                " margin: 2px 0px 2px 0px;\n"
                "}"
                ""
                "QTabBar::tab:selected, QTabBar::tab:hover {"
                " background: #6272a4; "
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

void MainWindow::add_editor()
{
  QFont jet_brains_mono = QFont("JetBrains Mono", 12);

  options = new QLineEdit();
  options->setText("-lineinfo, --gpu-architecture=compute_75");
  options->setFont(jet_brains_mono);

  QLineEdit *src_name = new QLineEdit();
  src_name->setText("source_1");
  src_name->setFont(jet_brains_mono);

  QObject::connect(options, &QLineEdit::textChanged, this, &MainWindow::reset_timer);

  cuda->setFont(jet_brains_mono);
  cuda->setAutoIndentation(true);
  cuda->setAutoParentheses(true);
  cuda->setTabReplace(true);
  cuda->setTabReplaceSize(2);

  ptx->setFont(jet_brains_mono);
  ptx->setReadOnly(true);

  QVBoxLayout *v_cuda_layout = new QVBoxLayout();
  v_cuda_layout->addWidget(src_name);
  v_cuda_layout->addWidget(cuda);

  QWidget *cuda_widget = new QWidget();
  cuda_widget->setLayout(v_cuda_layout);

  QVBoxLayout *v_ptx_layout = new QVBoxLayout();
  v_ptx_layout->addWidget(options);
  v_ptx_layout->addWidget(ptx);

  QWidget *ptx_widget = new QWidget();
  ptx_widget->setLayout(v_ptx_layout);

  QDockWidget *cuda_dock_widget = new QDockWidget("cuda", this);
  cuda_dock_widget->setWidget(cuda_widget);
  cuda_dock_widget->setFeatures(cuda_dock_widget->features() & ~QDockWidget::DockWidgetClosable);
  cuda_dock_widget->setAllowedAreas(Qt::AllDockWidgetAreas);
  addDockWidget(Qt::LeftDockWidgetArea, cuda_dock_widget);

  QDockWidget *ptx_dock_widget = new QDockWidget("ptx", this);
  ptx_dock_widget->setWidget(ptx_widget);
  ptx_dock_widget->setFeatures(ptx_dock_widget->features() & ~QDockWidget::DockWidgetClosable);
  ptx_dock_widget->setAllowedAreas(Qt::AllDockWidgetAreas);
  addDockWidget(Qt::RightDockWidgetArea, ptx_dock_widget);
}

KernelParameter get_int_param(const char *param_name, const QString &cuda_code)
{
  std::string result ("int ");
  result += param_name;
  result += " /* ";

  QString search_pattern = param_name;
  search_pattern += "\\s+=\\s*";

  auto pattern_pos = cuda_code.lastIndexOf(QRegularExpression(search_pattern));
  auto initializer_beg = cuda_code.indexOf('=', pattern_pos) + 1;
  auto initializer_end = cuda_code.indexOf("\n", initializer_beg);

  result += cuda_code.mid(initializer_beg, initializer_end - initializer_beg).toStdString();

  result += " */";

  return KernelParameter(result);
}

std::vector<std::string> split_params(QString params_str)
{
  std::vector<std::string> normalized_params;

  // TODO The whole function is a mess. I'm sorry if you are here.
  //      I'll fix it later
  while(true)
  {
    const bool stop = params_str.back() == ')';

    params_str = params_str.mid(0, params_str.size() - 1);

    if (stop)
    {
      break;
    }
  }

  QStringList lines = params_str.split('\n', Qt::SkipEmptyParts);

  for(QString& line: lines)
  {
    line = line.trimmed();

    if(line.back() == ',')
    {
      line = line.mid(0, line.size() - 1);
    }

    normalized_params.push_back(line.toStdString());
  }

  return normalized_params;
}

std::vector<KernelParameter> MainWindow::get_params()
{
  QString cuda_code = cuda->toPlainText();

  const int global_idx = cuda_code.lastIndexOf("__global__");
  const int params_start = cuda_code.indexOf('(', global_idx) + 1;
  const int params_len = cuda_code.indexOf('{', params_start) - params_start;

  std::vector<KernelParameter> result;

  for (auto &param: split_params(cuda_code.mid(params_start, params_len)))
  {
    result.emplace_back(param);
  }

  result.push_back(get_int_param("iterations", cuda_code));
  result.push_back(get_int_param("threads_in_block", cuda_code));
  result.push_back(get_int_param("blocks_in_grid", cuda_code));

  return result;
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

float min(const std::vector<float> &measurements)
{
  return *std::min_element(measurements.begin(), measurements.end());
}

float max(const std::vector<float> &measurements)
{
  return *std::max_element(measurements.begin(), measurements.end());
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

    QDockWidget *chart_widget = new QDockWidget("chart", this);
    chart_widget->setWidget(chart_view);
    chart_widget->setFeatures(chart_widget->features() & ~QDockWidget::DockWidgetClosable);
    addDockWidget(Qt::BottomDockWidgetArea, chart_widget);
    resizeDocks({chart_widget}, { 2 * height() / 3 }, Qt::Orientation::Vertical);
  }

  std::string ptx_code = ptx->toPlainText().toStdString();

  std::vector<float> elapsed_times = executor->execute(
    get_params(),
    ptx_code.c_str());

  const float min_time = min(elapsed_times);
  const float max_time = max(elapsed_times);

  min_elapsed = std::min(min_elapsed, min_time);
  max_elapsed = std::max(max_elapsed, max_time);

  for (auto &time: elapsed_times)
  {
    median_series.append(execution_id++, time);
  }

  chart->axes(Qt::Horizontal).back()->setRange(0, execution_id + 1);
  chart->axes(Qt::Vertical).back()->setRange(min_elapsed - min_elapsed / 4, max_elapsed + max_elapsed / 4);
}
